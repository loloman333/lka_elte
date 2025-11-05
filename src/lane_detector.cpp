// Lane detection pipeline (OpenCV).
// Notes:
// - Tune behavior via ld::LaneDetector::Params in the header.
// - Coordinates follow image conventions: origin at top-left, y grows downward.
// - Rendering colors and thicknesses are customizable in Params.

#include "lane_detector.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <sstream>
#include <fstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ld
{
    cv::Mat LaneDetector::processFrame(const cv::Mat &bgrFrame, PipelineDebug *dbg)
    {
        if (bgrFrame.empty())
            return cv::Mat();

        // Step 1: Convert to HLS.
        cv::Mat hls = convertToHLS(bgrFrame);

        // Step 1A: Apply CLAHE on L channel.
        // Improves contrast and stabilizes edges under varying lighting.
        std::vector<cv::Mat> ch;
        cv::split(hls, ch);
        cv::Mat L = ch[1];
        cv::Mat L_eq;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(params_.claheClipLimit, cv::Size(params_.claheTileSize, params_.claheTileSize));
        clahe->apply(L, L_eq);

        // Step 2: Edge detection.
        // Gaussian blur mitigates noise; Canny finds crisp edges.
        cv::Mat edges = detectEdges(L_eq);

        // Step 3: Apply ROI mask.
        // Keep lower roiKeepRatio of the frame, optionally cutting sides by roiAngleDeg.
        cv::Mat mask = roiMask(edges.size(), params_.roiKeepRatio, params_.roiAngleDeg);
        cv::Mat masked;
        cv::bitwise_and(edges, mask, masked);

        // Step 4: Detect Hough lines.
        // Raw Hough lines are drawn separately for debugging.
        std::vector<cv::Vec4i> hough_lines = detectLines(masked);
        cv::Mat hough_line_img;
        bgrFrame.copyTo(hough_line_img);
        drawLines(hough_line_img, hough_lines, params_.colorHoughRaw, params_.lineThickness);

        // Step 5: Pick seed lines and classify.
        // Raycast-based seeds -> optional RANSAC filter -> group agreeing lines.
        auto [seedLeft, seedRight] = pickFirstLines(hough_lines, bgrFrame.cols, bgrFrame.rows, params_.seedRayCount);
        seedLeft = ransacFilterSeedLines(seedLeft, bgrFrame.rows, params_.ransacIterations, params_.ransacInlierThreshPx);
        seedRight = ransacFilterSeedLines(seedRight, bgrFrame.rows, params_.ransacIterations, params_.ransacInlierThreshPx);
        auto [leftLines, rightLines] = findAgreeingLines(hough_lines, seedLeft, seedRight, params_.agreeSlopeTol, params_.agreeMaxEndpointDist);
        cv::Mat classifiedLines = bgrFrame.clone();
        drawLines(classifiedLines, leftLines, params_.colorLeftLines, params_.lineThickness);
        drawLines(classifiedLines, rightLines, params_.colorRightLines, params_.lineThickness);

        // Step 6: Build lane masks.
        // Sweep around each line segment perpendicularly and pick edge pixels.
        cv::Mat leftLaneMask = buildLaneMask(masked, leftLines, rightLines);
        cv::Mat rightLaneMask = buildLaneMask(masked, rightLines, leftLines);

        // Overlay masks in color to form a preview layer.
        cv::Mat lanePixels = cv::Mat::zeros(bgrFrame.size(), bgrFrame.type());
        lanePixels.setTo(params_.colorLeftLines, leftLaneMask);
        lanePixels.setTo(params_.colorRightLines, rightLaneMask);
        cv::addWeighted(lanePixels, 0.5, bgrFrame, 0.5, 0, lanePixels, CV_8U);

        // Step 7: Fit polynomials.
        // We fit x(y) = a*y^2 + b*y + c to account for mild curvature.
        cv::Mat leftLanePoints, rightLanePoints;
        cv::findNonZero(leftLaneMask, leftLanePoints);
        cv::findNonZero(rightLaneMask, rightLanePoints);

        cv::Mat leftLanePoly, rightLanePoly;
        if (leftLanePoints.rows > 0) leftLanePoly = fitPolynomial(leftLanePoints);
        if (rightLanePoints.rows > 0) rightLanePoly = fitPolynomial(rightLanePoints);

        // Draw raw fits for inspection.
        cv::Mat raw_poly_img = bgrFrame.clone();
        const int topY_dbg = static_cast<int>(std::round(bgrFrame.rows - params_.roiKeepRatio * bgrFrame.rows));
        const int bottomY_dbg = bgrFrame.rows - 1;
        if (!leftLanePoly.empty())
        {
            cv::Vec3d abc(leftLanePoly.at<double>(0, 0),
                          leftLanePoly.at<double>(1, 0),
                          leftLanePoly.at<double>(2, 0));
            drawQuadratic(raw_poly_img, abc, topY_dbg, bottomY_dbg, params_.colorLeftPoly, params_.polyThickness);
        }
        if (!rightLanePoly.empty())
        {
            cv::Vec3d abc(rightLanePoly.at<double>(0, 0),
                          rightLanePoly.at<double>(1, 0),
                          rightLanePoly.at<double>(2, 0));
            drawQuadratic(raw_poly_img, abc, topY_dbg, bottomY_dbg, params_.colorRightPoly, params_.polyThickness);
        }

        // Step 8: Temporal smoothing.
        // Blend current coefficients with a short, exponentially weighted history.
        cv::Mat leftLanePolySm, rightLanePolySm;
        smoothPolynomials(leftLanePoly, rightLanePoly, leftLanePolySm, rightLanePolySm);

        // Draw smoothed fits as final output.
        cv::Mat out = bgrFrame.clone();

        // Confidence estimation (more conservative).
        const cv::Mat leftPolyForEval = (!leftLanePolySm.empty() ? leftLanePolySm : leftLanePoly);
        const cv::Mat rightPolyForEval = (!rightLanePolySm.empty() ? rightLanePolySm : rightLanePoly);
        const double confLeft = estimateLaneConfidence(leftLaneMask, leftLines, leftPolyForEval, out.size(), /*isLeft=*/true);
        const double confRight = estimateLaneConfidence(rightLaneMask, rightLines, rightPolyForEval, out.size(), /*isLeft=*/false);
        const bool leftDetected = confLeft >= params_.confidenceThresh;
        const bool rightDetected = confRight >= params_.confidenceThresh;

        // CSV: one row per frame
        if (ensureCsvOpened())
        {
            if (!csvHeaderWritten_ && params_.csvWriteHeader && !params_.csvAppend)
                writeCsvHeader();
            writeCsvRow(frameId_, leftDetected, rightDetected, confLeft, confRight);
            ++frameId_;
        }

        // Draw final overlays:
        // - Solid colored poly if detected.
        // - Dashed gray poly (or dashed vertical fallback) if not detected.
        auto drawSide = [&](const cv::Mat &poly, bool detected, const cv::Scalar &solidColor, bool isLeft)
        {
            const int topY = topY_dbg;
            const int bottomY = bottomY_dbg;
            if (detected && !poly.empty())
            {
                cv::Vec3d abc(poly.at<double>(0, 0),
                              poly.at<double>(1, 0),
                              poly.at<double>(2, 0));
                drawQuadratic(out, abc, topY, bottomY, solidColor, params_.polyThickness);
            }
            else
            {
                if (!poly.empty())
                {
                    cv::Vec3d abc(poly.at<double>(0, 0),
                                  poly.at<double>(1, 0),
                                  poly.at<double>(2, 0));
                    drawQuadraticDashed(out, abc, topY, bottomY,
                                        params_.colorDashedUnknown, params_.polyThickness,
                                        params_.dashLenPx, params_.gapLenPx);
                }
                else
                {
                    // Fallback dashed vertical guide at 1/4 or 3/4 width.
                    int x = isLeft ? static_cast<int>(out.cols * 0.25) : static_cast<int>(out.cols * 0.75);
                    drawVerticalDashed(out, x, topY, bottomY,
                                       params_.colorDashedUnknown, params_.polyThickness,
                                       params_.dashLenPx, params_.gapLenPx);
                }
            }
        };

        drawSide(leftLanePolySm.empty() ? leftLanePoly : leftLanePolySm,
                 leftDetected, params_.colorLeftPoly, /*isLeft=*/true);
        drawSide(rightLanePolySm.empty() ? rightLanePoly : rightLanePolySm,
                 rightDetected, params_.colorRightPoly, /*isLeft=*/false);

        // HUD
        drawHUD(out, leftDetected, rightDetected, confLeft, confRight);

        if (dbg)
        {
            // Collect debug steps in order for UI viewers.
            dbg->steps.clear();
            auto pushStep = [&](const std::string &name, const cv::Mat &img)
            {
                if (!img.empty())
                    dbg->steps.push_back({name, img});
            };
            pushStep("01_preprocess", L_eq);
            pushStep("02_edges", edges);
            pushStep("03_apply_roi_mask", masked);
            pushStep("04_hough_lines", hough_line_img);
            pushStep("05_classify_lines", classifiedLines);
            pushStep("06_lane_pixels", lanePixels);
            pushStep("07_fit_polynoms_raw", raw_poly_img);
            pushStep("08_smooth_polynoms", out);
        }

        // Keep HighGUI responsive.
        cv::waitKey(1);

        return out;
    }

    cv::Mat LaneDetector::convertToHLS(const cv::Mat &bgr)
    {
        cv::Mat hls;
        cv::cvtColor(bgr, hls, cv::COLOR_BGR2HLS);
        return hls;
    }

    cv::Mat LaneDetector::detectEdges(const cv::Mat &gray)
    {
        cv::Mat blur;
        cv::GaussianBlur(gray, blur, cv::Size(params_.gaussKernel, params_.gaussKernel), 0);

        cv::Mat edges;
        cv::Canny(blur, edges, params_.cannyLow, params_.cannyHigh);

        return edges;
    }

    std::pair<cv::Vec4i, cv::Vec4i>
    LaneDetector::pickFirstLinesByBottom(const std::vector<cv::Vec4i> &lines, int imgWidth, int imgHeight, int minLen /*= 30*/, double minAbsSlope /*= 0.15*/)
    {
        double centerX = imgWidth * 0.5;

        bool haveLeft = false, haveRight = false;
        int bestLeftY = -1;      // track largest y (most bottom) for left
        int bestRightY = -1;     // track largest y (most bottom) for right
        double bestLeftX = -1e9; // tie-break: for left prefer larger x (more right)
        double bestRightX = 1e9; // tie-break: for right prefer smaller x (more left)
        cv::Vec4i bestLeftLine = cv::Vec4i(0, 0, 0, 0), bestRightLine = cv::Vec4i(0, 0, 0, 0);

        for (const auto &l : lines)
        {
            double dx = static_cast<double>(l[2] - l[0]);
            double dy = static_cast<double>(l[3] - l[1]);
            double len = std::hypot(dx, dy);
            if (len < minLen)
                continue;

            double slope;
            if (std::abs(dx) < 1e-6)
                slope = (dy > 0 ? 1e9 : -1e9);
            else
                slope = dy / dx;
            if (std::abs(slope) < minAbsSlope)
                continue;

            // use the line endpoint that is closest to the bottom (larger y)
            int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            int xBot, yBot;
            if (y1 >= y2)
            {
                xBot = x1;
                yBot = y1;
            }
            else
            {
                xBot = x2;
                yBot = y2;
            }

            if (slope < 0.0)
            { // left candidate (assuming y downwards)
                if (xBot < centerX)
                {
                    // prefer more bottom; tie-break: more right (larger xBot)
                    if (!haveLeft || yBot > bestLeftY || (yBot == bestLeftY && xBot > bestLeftX))
                    {
                        bestLeftY = yBot;
                        bestLeftX = xBot;
                        bestLeftLine = l;
                        haveLeft = true;
                    }
                }
            }
            else
            { // right candidate
                if (xBot > centerX)
                {
                    // prefer more bottom; tie-break: more left (smaller xBot)
                    if (!haveRight || yBot > bestRightY || (yBot == bestRightY && xBot < bestRightX))
                    {
                        bestRightY = yBot;
                        bestRightX = xBot;
                        bestRightLine = l;
                        haveRight = true;
                    }
                }
            }
        }

        // ensure default lines are valid Vec4i values; return the chosen best lines
        if (!haveLeft)
            bestLeftLine = cv::Vec4i(0, 0, 0, 0);
        if (!haveRight)
            bestRightLine = cv::Vec4i(0, 0, 0, 0);
        return {bestLeftLine, bestRightLine};
    }

    // Raycast-based picking: cast N horizontal scanlines across the ROI,
    // pick the nearest left/right intersection per scanline, deduplicate, and return.
    std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
    LaneDetector::pickFirstLines(const std::vector<cv::Vec4i> &lines,
                                 int imgWidth,
                                 int imgHeight,
                                 int numRays /*=3*/)
    {
        if (numRays < 1)
            numRays = 1;

        const double centerX = 0.5 * static_cast<double>(imgWidth);
        const int bottomY = imgHeight - 1;
        int topY = static_cast<int>(std::round(imgHeight - params_.roiKeepRatio * imgHeight));
        topY = std::max(0, std::min(topY, bottomY));
        const int roiH = bottomY - topY + 1;

        auto slopeOf = [](const cv::Vec4i &l)
        {
            const double dx = static_cast<double>(l[2] - l[0]);
            const double dy = static_cast<double>(l[3] - l[1]);
            if (std::abs(dx) < 1e-6)
                return std::numeric_limits<double>::infinity();
            return dy / dx;
        };

        std::vector<int> leftIdx, rightIdx;
        leftIdx.reserve(numRays);
        rightIdx.reserve(numRays);

        auto addUnique = [](std::vector<int> &vec, int idx)
        {
            if (std::find(vec.begin(), vec.end(), idx) == vec.end())
                vec.push_back(idx);
        };

        for (int r = 0; r < numRays; ++r)
        {
            const double t = (numRays == 1) ? 0.5 : static_cast<double>(r) / (numRays - 1);
            const int y = topY + static_cast<int>(std::round(t * (roiH - 1)));

            double bestLeftX = -1e18;
            int bestLeftI = -1;
            double bestRightX = 1e18;
            int bestRightI = -1;

            for (int i = 0; i < static_cast<int>(lines.size()); ++i)
            {
                const auto &l = lines[i];
                const double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];

                const double ymin = std::min(y1, y2), ymax = std::max(y1, y2);
                if (y < ymin - 1e-6 || y > ymax + 1e-6)
                    continue;

                const double dy = y2 - y1;
                if (std::abs(dy) < 1e-9)
                    continue; // skip near-horizontal segments

                const double xInt = x1 + (y - y1) * (x2 - x1) / dy;
                const double k = slopeOf(l);

                // Left: negative slope and intersection to the left of center
                if (k < 0.0 && xInt < centerX && xInt > bestLeftX)
                {
                    bestLeftX = xInt;
                    bestLeftI = i;
                }
                // Right: positive slope and intersection to the right of center
                if (k > 0.0 && xInt > centerX && xInt < bestRightX)
                {
                    bestRightX = xInt;
                    bestRightI = i;
                }
            }

            if (bestLeftI >= 0)
                addUnique(leftIdx, bestLeftI);
            if (bestRightI >= 0)
                addUnique(rightIdx, bestRightI);
        }

        std::vector<cv::Vec4i> left, right;
        left.reserve(leftIdx.size());
        right.reserve(rightIdx.size());
        for (int idx : leftIdx)
            left.push_back(lines[idx]);
        for (int idx : rightIdx)
            right.push_back(lines[idx]);

        return {left, right};
    }

    cv::Mat LaneDetector::roiMask(const cv::Size &size, double ratio, double angleDeg)
    {
        int w = size.width;
        int h = size.height;
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);

        // Clamp ratio to [0, 1].
        if (std::isnan(ratio))
            ratio = 0.0;
        ratio = std::max(0.0, std::min(1.0, ratio));

        // Normalize angle to [0, 180).
        if (std::isnan(angleDeg))
            angleDeg = 0.0;
        double angle = std::fmod(angleDeg, 180.0);
        if (angle < 0.0)
            angle += 180.0;

        // Interpret ratio as fraction of image height kept from the bottom.
        int bottomY = h - 1;
        int topY = static_cast<int>(std::round(h - ratio * h));

        if (topY < 0)
            topY = 0;
        if (topY > bottomY)
            topY = bottomY;

        // If angle is 0, fall back to a rectangle (no side cuts).
        if (angle <= 0.0 || topY >= bottomY)
        {
            cv::rectangle(mask, cv::Point(0, topY), cv::Point(w - 1, bottomY), cv::Scalar(255), cv::FILLED);
            return mask;
        }

        // Build a triangular ROI by cutting side triangles based on angle from vertical.
        const double cx = 0.5 * static_cast<double>(w - 1);
        const double dy = static_cast<double>(bottomY - topY);
        const double alpha = angle * CV_PI / 180.0;

        // Handle near-90° safely by pushing corners far off-image (avoids tan blow-ups).
        double dx;
        const double cosA = std::cos(alpha);
        if (std::abs(cosA) < 1e-9)
        {
            // horizontal rays -> make the base far off-image
            const double far = static_cast<double>(w + h) * 1000.0;
            dx = far;
        }
        else
        {
            dx = std::tan(alpha) * dy;
            if (!std::isfinite(dx))
            {
                const double far = static_cast<double>(w + h) * 1000.0;
                dx = (std::sin(alpha) >= 0 ? far : -far);
            }
        }

        // Do not clamp base intersections; allow polygon to extend off-image
        int xLeft = static_cast<int>(std::llround(cx - dx));
        int xRight = static_cast<int>(std::llround(cx + dx));
        if (xLeft > xRight)
            std::swap(xLeft, xRight);

        std::vector<cv::Point> tri{
            cv::Point(static_cast<int>(std::llround(cx)), topY),
            cv::Point(xRight, bottomY),
            cv::Point(xLeft, bottomY)};

        cv::fillConvexPoly(mask, tri.data(), static_cast<int>(tri.size()), cv::Scalar(255), cv::LINE_AA);

        return mask;
    }

    std::vector<cv::Vec4i> LaneDetector::detectLines(const cv::Mat &edges)
    {
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, params_.houghRho, params_.houghTheta, params_.houghThreshold,
                        params_.houghMinLineLength, params_.houghMaxLineGap);

        // Filter out nearly horizontal lines (lanes are typically slanted in image space).
        lines.erase(std::remove_if(lines.begin(), lines.end(),
                                   [&](const cv::Vec4i &l)
                                   {
                                       double dx = static_cast<double>(l[2] - l[0]);
                                       double dy = static_cast<double>(l[3] - l[1]);
                                       if (std::abs(dx) < 1e-6)
                                           return false; // Keep vertical lines.
                                       double slope = dy / dx;
                                       return std::abs(slope) < params_.minAbsSlopeFilter;
                                   }),
                    lines.end());

        return lines;
    }

    std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
    LaneDetector::findAgreeingLines(const std::vector<cv::Vec4i> &houghLines,
                                    const std::vector<cv::Vec4i> &seedLeft,
                                    const std::vector<cv::Vec4i> &seedRight,
                                    double slopeTol,
                                    double maxEndpointDist)
    {
        // Directly expand from the given seeds
        std::vector<cv::Vec4i> left = findAgreeingLinesForSide(houghLines, seedLeft, slopeTol, maxEndpointDist);
        std::vector<cv::Vec4i> right = findAgreeingLinesForSide(houghLines, seedRight, slopeTol, maxEndpointDist);
        return {left, right};
    }

    std::vector<cv::Vec4i>
    LaneDetector::findAgreeingLinesForSide(const std::vector<cv::Vec4i> &houghLines,
                                           const std::vector<cv::Vec4i> &seedLines,
                                           double slopeTol,
                                           double maxEndpointDist)
    {
        if (seedLines.empty())
            return {};
        std::vector<cv::Vec4i> out = seedLines; // Grow set transitively.

        // Compare slope similarity and endpoint proximity to expand clusters.
        auto slopeOf = [](const cv::Vec4i &l)
        {
            return static_cast<double>(l[3] - l[1]) / (static_cast<double>(l[2] - l[0]) + 1e-6);
        };
        auto closeByEndpoints = [&](const cv::Vec4i &a, const cv::Vec4i &b)
        {
            int ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
            int bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];
            double d1 = std::hypot(ax1 - bx1, ay1 - by1);
            double d2 = std::hypot(ax1 - bx2, ay1 - by2);
            double d3 = std::hypot(ax2 - bx1, ay2 - by1);
            double d4 = std::hypot(ax2 - bx2, ay2 - by2);
            return (d1 < maxEndpointDist || d2 < maxEndpointDist || d3 < maxEndpointDist || d4 < maxEndpointDist);
        };

        for (const auto &l : houghLines)
        {
            double k = slopeOf(l);
            // Compare against current accumulated lines; if it agrees with any, add it.
            for (const auto &ref : out)
            {
                double kr = slopeOf(ref);
                if (std::fabs(k - kr) < slopeTol && closeByEndpoints(l, ref))
                {
                    out.push_back(l);
                    break;
                }
            }
        }
        return out;
    }

    void LaneDetector::drawLine(cv::Mat &img, const cv::Vec4i &line, const cv::Scalar &color, int thickness)
    {
        cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), color, thickness, cv::LINE_AA);
    }

    void LaneDetector::drawLines(cv::Mat &img, const std::vector<cv::Vec4i> &lines, const cv::Scalar &color, int thickness)
    {
        for (const auto &line : lines)
        {
            cv::line(img, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), color, thickness, cv::LINE_AA);
        }
    }

    void LaneDetector::drawQuadratic(cv::Mat &img, const cv::Vec3d &abc, int topY, int bottomY, const cv::Scalar &color, int thickness)
    {
        // Draw x(y) curve from bottom to top to minimize visual gaps.
        auto xOfY = [&](double y) { return abc[0] * y * y + abc[1] * y + abc[2]; };

        // Clamp to image bounds.
        if (img.empty())
            return;
        topY = std::max(0, topY);
        bottomY = std::min(img.rows - 1, bottomY);
        if (bottomY < topY)
            std::swap(bottomY, topY);

        cv::Point prev(-1, -1);
        for (int y = bottomY; y >= topY; --y)
        {
            int xi = static_cast<int>(std::round(xOfY(y)));
            if (xi >= 0 && xi < img.cols)
            {
                cv::Point p(xi, y);
                if (prev.x >= 0)
                    cv::line(img, prev, p, color, thickness, cv::LINE_AA);
                prev = p;
            }
            else
            {
                // Reset when out of bounds.
                prev = {-1, -1};
            }
        }
    }

    void LaneDetector::drawQuadraticDashed(cv::Mat &img, const cv::Vec3d &abc, int topY, int bottomY,
                                           const cv::Scalar &color, int thickness, int dashLen, int gapLen)
    {
        if (img.empty()) return;
        topY = std::max(0, topY);
        bottomY = std::min(img.rows - 1, bottomY);
        if (bottomY < topY) std::swap(bottomY, topY);

        auto xOfY = [&](double y) { return abc[0] * y * y + abc[1] * y + abc[2]; };

        int run = 0;
        bool drawOn = true; // start with dash
        cv::Point prev(-1, -1);

        for (int y = bottomY; y >= topY; --y)
        {
            int xi = static_cast<int>(std::round(xOfY(y)));
            cv::Point p(xi, y);

            if (xi >= 0 && xi < img.cols)
            {
                if (prev.x >= 0 && drawOn)
                    cv::line(img, prev, p, color, thickness, cv::LINE_AA);
                prev = p;
                run++;
            }
            else
            {
                prev = {-1, -1};
                // Do not count out-of-bounds into dash run to keep cadence stable on-screen.
            }

            // toggle dash/gap based on vertical step count
            int segLen = drawOn ? dashLen : gapLen;
            if (run >= segLen)
            {
                drawOn = !drawOn;
                run = 0;
            }
        }
    }

    void LaneDetector::drawVerticalDashed(cv::Mat &img, int x, int topY, int bottomY,
                                          const cv::Scalar &color, int thickness, int dashLen, int gapLen)
    {
        if (img.empty()) return;
        x = std::max(0, std::min(img.cols - 1, x));
        topY = std::max(0, topY);
        bottomY = std::min(img.rows - 1, bottomY);
        if (bottomY < topY) std::swap(bottomY, topY);

        bool drawOn = true;
        int y = bottomY;
        while (y >= topY)
        {
            int seg = drawOn ? dashLen : gapLen;
            int y2 = std::max(topY, y - seg + 1);
            if (drawOn)
                cv::line(img, cv::Point(x, y), cv::Point(x, y2), color, thickness, cv::LINE_AA);
            drawOn = !drawOn;
            y = y2 - 1;
        }
    }

    cv::Mat LaneDetector::buildLaneMask(const cv::Mat &maskedEdges,
                                        const std::vector<cv::Vec4i> &leftLines,
                                        const std::vector<cv::Vec4i> &rightLines)
    {
        (void)rightLines; // Currently unused. Extend to fuse both sides if needed.

        cv::Mat leftLaneMask = cv::Mat::zeros(maskedEdges.size(), CV_8UC1);
        const int kernelHalf = params_.laneKernelHalf;
        const int sampleStep = params_.laneSampleStep;

        auto accumulateFromLines = [&](const std::vector<cv::Vec4i> &lines, cv::Mat &dstMask)
        {
            for (const auto &line : lines)
            {
                int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
                double dx = static_cast<double>(x2 - x1);
                double dy = static_cast<double>(y2 - y1);
                double length = std::hypot(dx, dy);
                if (length < 1.0)
                    continue;

                double ux = dx / length;
                double uy = dy / length;
                double px = -uy; // Perpendicular unit vector (x).
                double py = ux;  // Perpendicular unit vector (y).

                for (int s = 0; s <= static_cast<int>(length); s += sampleStep)
                {
                    int cx = static_cast<int>(std::round(x1 + ux * s));
                    int cy = static_cast<int>(std::round(y1 + uy * s));
                    if (cx < 0 || cx >= maskedEdges.cols || cy < 0 || cy >= maskedEdges.rows)
                        continue;

                    for (int off = -kernelHalf; off <= kernelHalf; ++off)
                    {
                        int sx = static_cast<int>(std::round(cx + px * off));
                        int sy = static_cast<int>(std::round(cy + py * off));
                        if (sx < 0 || sx >= maskedEdges.cols || sy < 0 || sy >= maskedEdges.rows)
                            continue;
                        if (maskedEdges.at<uchar>(sy, sx) > 0)
                        {
                            dstMask.at<uchar>(sy, sx) = 255;
                        }
                    }
                }
            }
        };

        // Accumulate from left-side lines.
        accumulateFromLines(leftLines, leftLaneMask);

        // Post-process to make the mask more continuous.
        if (!leftLaneMask.empty())
        {
            int dilateK = params_.laneDilateK;
            cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilateK, dilateK));
            cv::dilate(leftLaneMask, leftLaneMask, kern);
        }

        cv::Mat laneMask = leftLaneMask.clone();
        return laneMask;
    }

    bool LaneDetector::fitQuadraticXY(const std::vector<cv::Point> &points, cv::Vec3d &coeffs)
    {
        // Require a reasonable number of samples
        if (points.size() < 5)
            return false;

        const int n = static_cast<int>(points.size());
        cv::Mat A(n, 3, CV_64F), bx(n, 1, CV_64F);

        for (int i = 0; i < n; ++i)
        {
            const double y = static_cast<double>(points[i].y);
            A.at<double>(i, 0) = y * y;                             // y^2
            A.at<double>(i, 1) = y;                                 // y
            A.at<double>(i, 2) = 1.0;                               // 1
            bx.at<double>(i, 0) = static_cast<double>(points[i].x); // x
        }

        cv::Mat sol;
        if (!cv::solve(A, bx, sol, cv::DECOMP_SVD))
            return false;

        coeffs = cv::Vec3d(sol.at<double>(0, 0), sol.at<double>(1, 0), sol.at<double>(2, 0));
        return true;
    }

    cv::Mat LaneDetector::fitPolynomial(const cv::Mat &pointsMat)
    {
        if (pointsMat.empty())
            return cv::Mat();

        std::vector<cv::Point> pts;
        pts.reserve(pointsMat.rows);

        // Support common point matrix types produced by Mat::push_back(Point)
        int t = pointsMat.type();
        if (t == CV_32SC2)
        {
            for (int i = 0; i < pointsMat.rows; ++i)
            {
                const cv::Point &p = pointsMat.at<cv::Point>(i, 0);
                pts.push_back(p);
            }
        }
        else if (t == CV_32FC2)
        {
            for (int i = 0; i < pointsMat.rows; ++i)
            {
                const cv::Point2f &p = pointsMat.at<cv::Point2f>(i, 0);
                pts.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
            }
        }
        else if (t == CV_64FC2)
        {
            for (int i = 0; i < pointsMat.rows; ++i)
            {
                const cv::Point2d &p = pointsMat.at<cv::Point2d>(i, 0);
                pts.emplace_back(static_cast<int>(std::round(p.x)), static_cast<int>(std::round(p.y)));
            }
        }
        else
        {
            // Fallback: try to interpret as Nx2 CV_32S
            if (pointsMat.cols == 2 && pointsMat.type() == CV_32S)
            {
                for (int i = 0; i < pointsMat.rows; ++i)
                {
                    int x = pointsMat.at<int>(i, 0);
                    int y = pointsMat.at<int>(i, 1);
                    pts.emplace_back(x, y);
                }
            }
            else
            {
                return cv::Mat(); // unsupported layout
            }
        }

        cv::Vec3d abc;
        if (!fitQuadraticXY(pts, abc))
            return cv::Mat();

        cv::Mat coeffs(3, 1, CV_64F);
        coeffs.at<double>(0, 0) = abc[0]; // a
        coeffs.at<double>(1, 0) = abc[1]; // b
        coeffs.at<double>(2, 0) = abc[2]; // c
        return coeffs;
    }

    void LaneDetector::smoothPolynomials(const cv::Mat &leftLanePoly,
                                         const cv::Mat &rightLanePoly,
                                         cv::Mat &leftLanePolySm,
                                         cv::Mat &rightLanePolySm)
    {
        auto matToVec3d = [](const cv::Mat &m) -> cv::Vec3d {
            return cv::Vec3d(m.at<double>(0,0), m.at<double>(1,0), m.at<double>(2,0));
        };
        auto vec3dToMat = [](const cv::Vec3d &v) -> cv::Mat {
            cv::Mat m(3,1,CV_64F);
            m.at<double>(0,0) = v[0];
            m.at<double>(1,0) = v[1];
            m.at<double>(2,0) = v[2];
            return m;
        };

        auto pushWithLimit = [](std::deque<cv::Vec3d> &hist, const cv::Vec3d &val, int maxLen) {
            if (hist.size() >= static_cast<size_t>(maxLen)) hist.pop_front();
            hist.push_back(val);
        };

        auto smoothFromHist = [&](const std::deque<cv::Vec3d> &hist) -> cv::Mat {
            if (hist.empty()) return cv::Mat();
            cv::Vec3d acc(0.0, 0.0, 0.0);
            double wsum = 0.0;
            const int n = static_cast<int>(hist.size());
            for (int idx = 0; idx < n; ++idx)
            {
                const cv::Vec3d &v = hist[n - 1 - idx];
                const double w = std::pow(polySmoothAlpha_, static_cast<double>(idx));
                acc += w * v;
                wsum += w;
            }
            if (wsum <= 0.0) return cv::Mat();
            cv::Vec3d sm = acc * (1.0 / wsum);
            return vec3dToMat(sm);
        };

        if (!leftLanePoly.empty())
            pushWithLimit(histLeft_, matToVec3d(leftLanePoly), polyHistoryLen_);
        if (!rightLanePoly.empty())
            pushWithLimit(histRight_, matToVec3d(rightLanePoly), polyHistoryLen_);

        leftLanePolySm = smoothFromHist(histLeft_);
        rightLanePolySm = smoothFromHist(histRight_);
    }

    double LaneDetector::estimateLaneConfidence(const cv::Mat &laneMask,
                                                const std::vector<cv::Vec4i> &agreeingLines,
                                                const cv::Mat &poly,
                                                const cv::Size &imgSize,
                                                bool isLeft)
    {
        auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };

        // 1) Global mask density (saturates around ~1% of image area).
        double fMask = 0.0;
        if (!laneMask.empty())
        {
            const int nz = cv::countNonZero(laneMask);
            const double norm = 0.01 * static_cast<double>(imgSize.area()); // ~1% of pixels
            fMask = clamp01(nz / std::max(1.0, norm));
        }

        // 2) Number of agreeing Hough lines (saturates quickly).
        double fLines = clamp01(static_cast<double>(agreeingLines.size()) / 6.0);

        // 3) Slope consistency: lower stddev -> higher confidence.
        double fSlope = 0.0;
        if (agreeingLines.size() >= 2)
        {
            std::vector<double> slopes;
            slopes.reserve(agreeingLines.size());
            for (const auto &l : agreeingLines)
            {
                double dx = static_cast<double>(l[2] - l[0]);
                double dy = static_cast<double>(l[3] - l[1]);
                if (std::abs(dx) < 1e-6)
                    continue; // treat near-vertical as very large slope; skip to avoid blow-up
                slopes.push_back(dy / dx);
            }
            if (slopes.size() >= 2)
            {
                double mean = 0.0;
                for (double s : slopes) mean += s;
                mean /= static_cast<double>(slopes.size());
                double var = 0.0;
                for (double s : slopes) var += (s - mean) * (s - mean);
                var /= static_cast<double>(slopes.size() - 1);
                double sd = std::sqrt(std::max(0.0, var));
                fSlope = std::exp(-sd / std::max(1e-6, params_.confSlopeStdRef));
                fSlope = clamp01(fSlope);
            }
        }

        // Poly unpack helpers.
        auto hasPoly = [&]() { return !poly.empty() && poly.rows >= 3 && poly.cols >= 1; };
        auto polyABC = [&]() -> cv::Vec3d {
            return cv::Vec3d(poly.at<double>(0,0), poly.at<double>(1,0), poly.at<double>(2,0));
        };

        // ROI vertical range.
        int bottomY = imgSize.height - 1;
        int topY = static_cast<int>(std::round(imgSize.height - params_.roiKeepRatio * imgSize.height));
        topY = std::max(0, std::min(topY, bottomY));

        // 4) Coverage along the curve: fraction of sampled rows with mask support near x(y).
        double fCoverage = 0.0;
        if (hasPoly() && !laneMask.empty())
        {
            const cv::Vec3d abc = polyABC();
            auto xOfY = [&](double y) { return abc[0] * y * y + abc[1] * y + abc[2]; };

            int valid = 0, supported = 0;
            for (int y = bottomY; y >= topY; y -= std::max(1, params_.confSupportSampleStep))
            {
                int xi = static_cast<int>(std::llround(xOfY(static_cast<double>(y))));
                if (xi < 0 || xi >= laneMask.cols) continue;
                ++valid;

                int half = std::max(0, params_.confSupportBandHalf);
                int xl = std::max(0, xi - half);
                int xr = std::min(laneMask.cols - 1, xi + half);
                const uchar* row = laneMask.ptr<uchar>(y);
                bool any = false;
                for (int x = xl; x <= xr; ++x)
                {
                    if (row[x] != 0) { any = true; break; }
                }
                if (any) ++supported;
            }
            fCoverage = (valid > 0) ? static_cast<double>(supported) / static_cast<double>(valid) : 0.0;
            fCoverage = clamp01(fCoverage);
        }

        // 5) Temporal stability vs. last smoothed polynomial for this side.
        double fStability = 0.0;
        if (hasPoly())
        {
            const auto &hist = isLeft ? histLeft_ : histRight_;
            if (!hist.empty())
            {
                cv::Vec3d cur = polyABC();
                cv::Vec3d prev = hist.back();
                cv::Vec3d d = cur - prev;

                // Approximate pixel deviation across ROI with three samples.
                auto xOf = [&](const cv::Vec3d &c, double y) { return c[0]*y*y + c[1]*y + c[2]; };
                double yT = static_cast<double>(topY);
                double yM = 0.5 * (static_cast<double>(topY) + static_cast<double>(bottomY));
                double yB = static_cast<double>(bottomY);
                double eT = std::abs(xOf(d, yT));
                double eM = std::abs(xOf(d, yM));
                double eB = std::abs(xOf(d, yB));
                double ePx = (eT + eM + eB) / 3.0;

                fStability = std::exp(-ePx / std::max(1e-6, params_.confStabilityRefPx));
                fStability = clamp01(fStability);
            }
        }

        // 6) Curvature penalty: large |a| reduces confidence.
        double fCurvature = 0.0;
        if (hasPoly())
        {
            double a = std::abs(poly.at<double>(0,0));
            fCurvature = std::exp(-a / std::max(1e-12, params_.confCurvatureRef));
            fCurvature = clamp01(fCurvature);
        }

        // Weighted combination (normalize weights sum if needed).
        const double wSum =
            params_.confWMask + params_.confWLines + params_.confWSlope +
            params_.confWCoverage + params_.confWStability + params_.confWCurvature;
        const double invWSum = (wSum > 0.0) ? (1.0 / wSum) : 1.0;

        double conf =
            params_.confWMask * fMask +
            params_.confWLines * fLines +
            params_.confWSlope * fSlope +
            params_.confWCoverage * fCoverage +
            params_.confWStability * fStability +
            params_.confWCurvature * fCurvature;

        conf *= invWSum;
        conf += 0.1;
        return clamp01(conf);
    }

    void LaneDetector::drawHUD(cv::Mat &img,
                               bool leftDetected, bool rightDetected,
                               double confLeft, double confRight)
    {
        if (img.empty()) return;

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "Left: " << (leftDetected ? "YES" : "NO")
            <<  " Conf: " << confLeft
            << " | Right: " << (rightDetected ? "YES" : "NO")
            << " Conf: " << confRight;

        const std::string text = oss.str();
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = params_.hudFontScale;
        const int thick = params_.hudThickness;

        int baseline = 0;
        cv::Size sz = cv::getTextSize(text, font, scale, thick, &baseline);
        cv::Point org(10, 10 + sz.height);

        // Semi-transparent background box
        const int pad = std::max(0, params_.hudPadding);
        cv::Point tl(std::max(0, org.x - pad), std::max(0, org.y - sz.height - pad));
        cv::Point br(std::min(img.cols - 1, org.x + sz.width + pad), std::min(img.rows - 1, org.y + baseline + pad));
        cv::Rect box(tl, br);

        cv::Mat overlay = img.clone();
        cv::rectangle(overlay, box, params_.hudBgColor, cv::FILLED);
        const double alpha = std::max(0.0, std::min(1.0, params_.hudBgAlpha));
        cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);

        // Foreground text (no shadow)
        cv::putText(img, text, org, font, scale, params_.hudTextColor, thick, cv::LINE_AA);
    }

    // CSV helpers
    bool LaneDetector::ensureCsvOpened()
    {
        if (params_.csvOutputPath.empty())
            return false;
        if (csv_.is_open())
            return true;

        std::ios::openmode mode = std::ios::out;
        mode |= params_.csvAppend ? std::ios::app : std::ios::trunc;
        csv_.open(params_.csvOutputPath, mode);
        csv_.setf(std::ios::fixed);
        csv_.precision(2);

        // If appending, assume header already exists (avoid duplicate headers).
        if (params_.csvAppend) csvHeaderWritten_ = true;

        return csv_.good();
    }

    void LaneDetector::writeCsvHeader()
    {
        if (!csv_.is_open() || csvHeaderWritten_) return;
        csv_ << "frame_id,left_detected,right_detected,left_conf,right_conf\n";
        csvHeaderWritten_ = true;
        csv_.flush();
    }

    void LaneDetector::writeCsvRow(int frameId, bool leftDetected, bool rightDetected, double confLeft, double confRight)
    {
        if (!csv_.is_open()) return;
        csv_ << frameId << ','
             << (leftDetected ? 1 : 0) << ','
             << (rightDetected ? 1 : 0) << ','
             << confLeft << ','
             << confRight << '\n';
        csv_.flush();
    }

    // RANSAC-like filtering of seed lines using x = m*y + c.
    // Deterministic variant: test each seed as hypothesis and keep largest inlier set.
    std::vector<cv::Vec4i>
    LaneDetector::ransacFilterSeedLines(const std::vector<cv::Vec4i> &seeds,
                                        int imgHeight,
                                        int iterations,
                                        double inlierThreshPx)
    {
        (void)imgHeight;
        (void)iterations;

        if (seeds.size() <= 2) return seeds;

        auto lineToMC = [](const cv::Vec4i &l, double &m, double &c) -> bool
        {
            const double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            const double dy = y2 - y1;
            if (std::abs(dy) < 1e-6) return false; // avoid horizontal/degenerate
            m = (x2 - x1) / dy; // x = m*y + c
            c = x1 - m * y1;
            return std::isfinite(m) && std::isfinite(c);
        };
        auto residual = [&](double m, double c, const cv::Vec4i &l) -> double {
            const double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            const double r1 = std::abs(x1 - (m * y1 + c));
            const double r2 = std::abs(x2 - (m * y2 + c));
            return 0.5 * (r1 + r2);
        };

        std::vector<int> bestInliers;
        bestInliers.reserve(seeds.size());

        // Deterministic RANSAC: try each seed as hypothesis.
        for (int h = 0; h < static_cast<int>(seeds.size()); ++h) {
            double m = 0.0, c = 0.0;
            if (!lineToMC(seeds[h], m, c)) continue;

            std::vector<int> cur;
            cur.reserve(seeds.size());
            for (int i = 0; i < static_cast<int>(seeds.size()); ++i) {
                if (residual(m, c, seeds[i]) <= inlierThreshPx) cur.push_back(i);
            }
            if (cur.size() > bestInliers.size()) bestInliers.swap(cur);
        }

        if (bestInliers.empty()) return seeds;

        std::vector<cv::Vec4i> filtered;
        filtered.reserve(bestInliers.size());
        for (int idx : bestInliers) filtered.push_back(seeds[idx]);
        return filtered;
    }

} // namespace ld
