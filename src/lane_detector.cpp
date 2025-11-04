#include "lane_detector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <deque>

namespace ld
{
    // Temporal smoothing history for polynomial coefficients
    static double g_polySmoothAlpha = 0.9; // [0..1], lower -> stronger smoothing over history
    static int g_polyHistoryLen = 10;      // history length
    static std::deque<cv::Vec3d> g_histLeft;
    static std::deque<cv::Vec3d> g_histRight;

    // Smooth polynomial coefficients over a history buffer
    static void smoothPolynomials(const cv::Mat &leftLanePoly,
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
            // Exponentially decayed weights favoring newest samples:
            // newest weight = alpha^0 = 1, previous = alpha^1, ...
            cv::Vec3d acc(0.0, 0.0, 0.0);
            double wsum = 0.0;
            const int n = static_cast<int>(hist.size());
            for (int idx = 0; idx < n; ++idx)
            {
                // access from newest backwards
                const cv::Vec3d &v = hist[n - 1 - idx];
                const double w = std::pow(g_polySmoothAlpha, static_cast<double>(idx));
                acc += w * v;
                wsum += w;
            }
            if (wsum <= 0.0) return cv::Mat();
            cv::Vec3d sm = acc * (1.0 / wsum);
            return vec3dToMat(sm);
        };

        // Update left history if current available
        if (!leftLanePoly.empty())
            pushWithLimit(g_histLeft, matToVec3d(leftLanePoly), g_polyHistoryLen);
        // Update right history if current available
        if (!rightLanePoly.empty())
            pushWithLimit(g_histRight, matToVec3d(rightLanePoly), g_polyHistoryLen);

        // Produce smoothed outputs from history (falls back to older frames if current missing)
        leftLanePolySm = smoothFromHist(g_histLeft);
        rightLanePolySm = smoothFromHist(g_histRight);
    }

    cv::Mat LaneDetector::processFrame(const cv::Mat &bgrFrame, PipelineDebug *dbg)
    {
        if (bgrFrame.empty())
            return cv::Mat();

        // 1) HLS conversion
        cv::Mat hls = convertToHLS(bgrFrame);

        // 1.A) CLAHE on L channel
        std::vector<cv::Mat> ch;
        cv::split(hls, ch);
        cv::Mat L = ch[1];
        cv::Mat L_eq;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(L, L_eq);

        // 2) Edge detection
        cv::Mat edges = detectEdges(L_eq);

        // 3) ROI mask
        cv::Mat mask = roiMask(edges.size(), 0.55, 110.0);
        cv::Mat masked;
        cv::bitwise_and(edges, mask, masked);

        // 4) Hough lines
        std::vector<cv::Vec4i> hough_lines = detectLines(masked);
        cv::Mat hough_line_img;
        bgrFrame.copyTo(hough_line_img);
        drawLines(hough_line_img, hough_lines, cv::Scalar(0, 0, 255), 2);

        // 5) Seed lines and classification
        auto [seedLeft, seedRight] = pickFirstLines(hough_lines, bgrFrame.cols, bgrFrame.rows, 15);
        // RANSAC filter seeds (x = m*y + c)
        seedLeft = ransacFilterSeedLines(seedLeft, bgrFrame.rows, 50, 10.0);
        seedRight = ransacFilterSeedLines(seedRight, bgrFrame.rows, 50, 10.0);
        auto [leftLines, rightLines] = findAgreeingLines(hough_lines, seedLeft, seedRight);
        cv::Mat classifiedLines = bgrFrame.clone();
        drawLines(classifiedLines, leftLines, cv::Scalar(255, 0, 0), 2);
        drawLines(classifiedLines, rightLines, cv::Scalar(255, 255, 0), 2);

        // 6) Build lane masks
        cv::Mat leftLaneMask = buildLaneMask(masked, leftLines, rightLines);
        cv::Mat rightLaneMask = buildLaneMask(masked, rightLines, leftLines);

        cv::Mat lanePixels = cv::Mat::zeros(bgrFrame.size(), bgrFrame.type());
        lanePixels.setTo(cv::Scalar(255, 0, 0), leftLaneMask);
        lanePixels.setTo(cv::Scalar(255, 255, 0), rightLaneMask);
        cv::addWeighted(lanePixels, 0.5, bgrFrame, 0.5, 0, lanePixels, CV_8U);

        // 7) Fit polynomials
        cv::Mat leftLanePoints, rightLanePoints;
        cv::findNonZero(leftLaneMask, leftLanePoints);   // Nx1, CV_32SC2
        cv::findNonZero(rightLaneMask, rightLanePoints); // Nx1, CV_32SC2

        cv::Mat leftLanePoly, rightLanePoly;
        if (leftLanePoints.rows > 0)
        {
            leftLanePoly = fitPolynomial(leftLanePoints);
        }
        if (rightLanePoints.rows > 0)
        {
            rightLanePoly = fitPolynomial(rightLanePoints);
        }

        cv::Mat raw_poly_img = bgrFrame.clone();
        const int topY_dbg = static_cast<int>(params_.roiTopY * bgrFrame.rows);
        const int bottomY_dbg = bgrFrame.rows - 1;
        if (!leftLanePoly.empty())
        {
            cv::Vec3d abc(leftLanePoly.at<double>(0, 0),
                          leftLanePoly.at<double>(1, 0),
                          leftLanePoly.at<double>(2, 0));
            drawQuadratic(raw_poly_img, abc, topY_dbg, bottomY_dbg, cv::Scalar(0, 0, 255), 3); // red
        }
        if (!rightLanePoly.empty())
        {
            cv::Vec3d abc(rightLanePoly.at<double>(0, 0),
                          rightLanePoly.at<double>(1, 0),
                          rightLanePoly.at<double>(2, 0));
            drawQuadratic(raw_poly_img, abc, topY_dbg, bottomY_dbg, cv::Scalar(0, 255, 0), 3); // green
        }

        // 8) Temporal smoothing
        cv::Mat leftLanePolySm, rightLanePolySm;
        smoothPolynomials(leftLanePoly, rightLanePoly, leftLanePolySm, rightLanePolySm);

        cv::Mat out = bgrFrame.clone();
        if (!leftLanePolySm.empty())
        {
            cv::Vec3d abc(leftLanePolySm.at<double>(0, 0),
                          leftLanePolySm.at<double>(1, 0),
                          leftLanePolySm.at<double>(2, 0));
            drawQuadratic(out, abc, topY_dbg, bottomY_dbg, cv::Scalar(0, 0, 255), 3); // red
        }
        if (!rightLanePolySm.empty())
        {
            cv::Vec3d abc(rightLanePolySm.at<double>(0, 0),
                          rightLanePolySm.at<double>(1, 0),
                          rightLanePolySm.at<double>(2, 0));
            drawQuadratic(out, abc, topY_dbg, bottomY_dbg, cv::Scalar(0, 255, 0), 3); // green
        }

        if (dbg)
        {
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

        // Keep HighGUI responsive
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

    // Raycast-based picking returning lists for left and right
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
        int topY = static_cast<int>(params_.roiTopY * imgHeight);
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

        // clamp ratio to [0,1]
        if (std::isnan(ratio))
            ratio = 0.0;
        ratio = std::max(0.0, std::min(1.0, ratio));

        // Accept angles > 90°. Normalize to [0, 180) for symmetry around vertical.
        if (std::isnan(angleDeg))
            angleDeg = 0.0;
        double angle = std::fmod(angleDeg, 180.0);
        if (angle < 0.0)
            angle += 180.0;

        // Interpret ratio as fraction of image height to keep from the bottom.
        int bottomY = h - 1;
        int topY = static_cast<int>(std::round(h - ratio * h));

        if (topY < 0)
            topY = 0;
        if (topY > bottomY)
            topY = bottomY;

        // If angle is 0, fall back to simple rectangle (no side cuts)
        if (angle <= 0.0 || topY >= bottomY)
        {
            cv::rectangle(mask, cv::Point(0, topY), cv::Point(w - 1, bottomY), cv::Scalar(255), cv::FILLED);
            return mask;
        }

        // Build a triangular ROI by cutting side triangles based on angle from vertical.
        const double cx = 0.5 * static_cast<double>(w - 1);
        const double dy = static_cast<double>(bottomY - topY);
        const double alpha = angle * CV_PI / 180.0;

        // Handle near-90° safely
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

        // Do not clamp base intersections; let polygon extend off-image
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

        // filter out nearly horizontal lines
        lines.erase(std::remove_if(lines.begin(), lines.end(),
                                   [](const cv::Vec4i &l)
                                   {
                                       double dx = static_cast<double>(l[2] - l[0]);
                                       double dy = static_cast<double>(l[3] - l[1]);
                                       if (std::abs(dx) < 1e-6)
                                           return false; // keep vertical lines
                                       double slope = dy / dx;
                                       return std::abs(slope) < 0.3; // filter near-horizontal
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
        std::vector<cv::Vec4i> out = seedLines; // grow set transitively

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
            // compare against current accumulated lines; if agrees with any, add it
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
        auto xOfY = [&](double y)
        { return abc[0] * y * y + abc[1] * y + abc[2]; };

        // Clamp to image bounds
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
                prev = {-1, -1};
            }
        }
    }

    cv::Mat LaneDetector::buildLaneMask(const cv::Mat &maskedEdges,
                                        const std::vector<cv::Vec4i> &leftLines,
                                        const std::vector<cv::Vec4i> &rightLines)
    {
        (void)rightLines; // unused by design; mask is built from the provided 'lines' set

        cv::Mat leftLaneMask = cv::Mat::zeros(maskedEdges.size(), CV_8UC1);
        const int kernelHalf = 7;
        const int sampleStep = 2;

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
                double px = -uy; // perpendicular
                double py = ux;

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

        accumulateFromLines(leftLines, leftLaneMask);

        if (!leftLaneMask.empty())
        {
            int dilateK = 5;
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

    // RANSAC-like filtering of seed lines using x = m*y + c model
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

        // Deterministic "basic RANSAC": try each seed as hypothesis
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
