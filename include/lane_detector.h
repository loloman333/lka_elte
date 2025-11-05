#pragma once

#include <deque>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <fstream>

namespace ld
{
    /// Pipeline step output for debugging UIs.
    struct DebugStep
    {
        std::string name;
        cv::Mat image;
    };

    /// Collects labeled debug frames in processing order.
    struct PipelineDebug
    {
        // Ordered list of debug step images.
        std::vector<DebugStep> steps;
    };

    /// Lane detection pipeline working on BGR frames.
    /// Configure behavior via Params; all visual styles and thresholds live there.
    class LaneDetector
    {
    public:
        // Tuning parameters.
        struct Params
        {
            // Edge detection parameters (Canny and Gaussian blur).
            int cannyLow = 50;
            int cannyHigh = 120;
            int gaussKernel = 5;

            // Hough Transform parameters for line segments.
            double houghRho = 1.0;
            double houghTheta = CV_PI / 180.0;
            int houghThreshold = 20;
            double houghMinLineLength = 100.0;
            double houghMaxLineGap = 100.0;

            // ROI parameters: keep a trapezoid-like area near the bottom.
            double roiKeepRatio = 0.55; // Fraction of height kept from bottom.
            double roiAngleDeg = 110;   // Angle from vertical for side cuts.

            // CLAHE parameters to stabilize luminance before edges.
            double claheClipLimit = 2.0;
            int claheTileSize = 8;

            // RANSAC seed filtering to remove outlier Hough seeds.
            int ransacIterations = 50;
            double ransacInlierThreshPx = 10.0;

            // Seed picking (raycast count in ROI).
            int seedRayCount = 15;

            // Line agreement grouping thresholds.
            double agreeSlopeTol = 0.15;
            double agreeMaxEndpointDist = 8.0;

            // Hough line post-filter (reject near-horizontal).
            double minAbsSlopeFilter = 0.3;

            // Rendering styles.
            int lineThickness = 2;
            int polyThickness = 3;
            cv::Scalar colorHoughRaw = cv::Scalar(0, 0, 255);      // Red lines over masked edges.
            cv::Scalar colorLeftLines = cv::Scalar(0, 255, 0);     // Green for left lane clusters.
            cv::Scalar colorRightLines = cv::Scalar(255, 0, 0);    // Blue for right lane clusters.
            cv::Scalar colorLeftPoly = cv::Scalar(0, 255, 0);      // Left = green polyline
            cv::Scalar colorRightPoly = cv::Scalar(255, 0, 0);     // Right = blue polyline
            cv::Scalar colorDashedUnknown = cv::Scalar(160, 160, 160); // Dashed gray when not detected

            // Lane mask shaping around detected lines.
            int laneKernelHalf = 7;  // Half-width of the perpendicular sweep.
            int laneSampleStep = 2;  // Step along the line when sampling.
            int laneDilateK = 5;     // Final dilation to connect sparse pixels.

            // Confidence and dashed styling.
            double confidenceThresh = 0.35; // Liberal threshold for YES/NO
            int dashLenPx = 12;
            int gapLenPx = 8;

            // HUD styling.
            cv::Scalar hudTextColor = cv::Scalar(255, 255, 255);
            cv::Scalar hudShadowColor = cv::Scalar(0, 0, 0);
            double hudFontScale = 0.6;
            int hudThickness = 1;
            cv::Scalar hudBgColor = cv::Scalar(0, 0, 0); // semi-transparent background box
            double hudBgAlpha = 0.4;                     // [0..1]
            int hudPadding = 6;                          // px

            // Confidence estimator tuning (lightweight).
            int confSupportBandHalf = 6;      // Half-band around x(y) to search mask support.
            int confSupportSampleStep = 5;    // Row sampling step for coverage/stability checks.
            double confSlopeStdRef = 0.25;    // Reference slope std for exp decay mapping.
            double confStabilityRefPx = 10.0; // ~10px deviation yields ~e^-1 drop.
            double confCurvatureRef = 2e-6;   // Reference |a| for curvature penalty (x = a*y^2 + ...).

            // Feature weights (sum ~= 1.0).
            double confWMask = 0.35;
            double confWLines = 0.25;
            double confWSlope = 0.25;
            double confWCoverage = 0.40;
            double confWStability = 0.20;
            double confWCurvature = 0.15;

            // CSV export (empty path disables).
            std::string csvOutputPath = "output/metrics.csv"; // e.g. "/tmp/lane_metrics.csv"
            bool csvAppend = false;         // append if true, otherwise truncate
            bool csvWriteHeader = true;     // write header when not appending
        };

        LaneDetector() = default;
        explicit LaneDetector(const Params &p) : params_(p) {}

        // Process one BGR frame and return an annotated BGR frame.
        // Optionally fills PipelineDebug with intermediate visualization steps.
        cv::Mat processFrame(const cv::Mat &bgrFrame, PipelineDebug *dbg = nullptr);

    private:
        Params params_{};

        // Color space conversion helpers.
        cv::Mat convertToHLS(const cv::Mat &bgr);
        cv::Mat detectEdges(const cv::Mat &bgrOrGray);

        // ROI mask. Angle is from vertical at the center-top cut; 0 keeps rectangular ROI.
        cv::Mat roiMask(const cv::Size &size, double ratio = 0.66, double angleDeg = 0.0);

        // Line detection and selection.
        std::vector<cv::Vec4i> detectLines(const cv::Mat &edges);

        // Simpler baseline line picker (kept for comparison).
        std::pair<cv::Vec4i, cv::Vec4i>
        pickFirstLinesByBottom(const std::vector<cv::Vec4i> &lines, int imgWidth, int imgHeight, int minLen = 30, double minAbsSlope = 0.15);

        // Raycast-based seed picker (left/right lists).
        std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
        pickFirstLines(const std::vector<cv::Vec4i> &lines, int imgWidth, int imgHeight, int numRays = 3);

        // RANSAC-like seed filter to remove outliers (x = m*y + c model).
        std::vector<cv::Vec4i>
        ransacFilterSeedLines(const std::vector<cv::Vec4i> &seeds, int imgHeight, int iterations = 50, double inlierThreshPx = 10.0);

        // Polynomial fitting and drawing.
        bool fitQuadraticXY(const std::vector<cv::Point> &points, cv::Vec3d &coeffs); // x = a*y^2 + b*y + c.
        void drawQuadratic(cv::Mat &img, const cv::Vec3d &abc, int topY, int bottomY, const cv::Scalar &color, int thickness = 3);
        void drawQuadraticDashed(cv::Mat &img, const cv::Vec3d &abc, int topY, int bottomY,
                                 const cv::Scalar &color, int thickness, int dashLen, int gapLen);
        void drawVerticalDashed(cv::Mat &img, int x, int topY, int bottomY,
                                const cv::Scalar &color, int thickness, int dashLen, int gapLen);
        cv::Mat fitPolynomial(const cv::Mat &points); // Returns [a, b, c]^T for x = a*y^2 + b*y + c.

        // Group Hough lines that agree with seed left/right lines.
        std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
        findAgreeingLines(const std::vector<cv::Vec4i> &houghLines,
                          const std::vector<cv::Vec4i> &seedLeft,
                          const std::vector<cv::Vec4i> &seedRight,
                          double slopeTol = 0.2,
                          double maxEndpointDist = 10.0);

        // Helper used by findAgreeingLines for one side.
        std::vector<cv::Vec4i>
        findAgreeingLinesForSide(const std::vector<cv::Vec4i> &houghLines,
                                 const std::vector<cv::Vec4i> &seedLines,
                                 double slopeTol,
                                 double maxEndpointDist);

        // Basic drawing helpers.
        void drawLine(cv::Mat &img, const cv::Vec4i &line, const cv::Scalar &color, int thickness = 1);
        void drawLines(cv::Mat &img, const std::vector<cv::Vec4i> &lines, const cv::Scalar &color, int thickness = 1);

        // Build a lane mask from edges and selected lines (currently uses leftLines only).
        cv::Mat buildLaneMask(const cv::Mat &maskedEdges,
                              const std::vector<cv::Vec4i> &leftLines,
                              const std::vector<cv::Vec4i> &rightLines);

        // Temporal smoothing of fitted polynomials with history.
        void smoothPolynomials(const cv::Mat &leftLanePoly,
                               const cv::Mat &rightLanePoly,
                               cv::Mat &leftLanePolySm,
                               cv::Mat &rightLanePolySm);

        // Temporal smoothing parameters and history.
        double polySmoothAlpha_ = 0.95; // [0..1], lower means stronger smoothing over history.
        int polyHistoryLen_ = 20;      // History length.
        std::deque<cv::Vec3d> histLeft_;
        std::deque<cv::Vec3d> histRight_;

        // Confidence estimation [0..1] for a side.
        double estimateLaneConfidence(const cv::Mat &laneMask,
                                      const std::vector<cv::Vec4i> &agreeingLines,
                                      const cv::Mat &poly,
                                      const cv::Size &imgSize,
                                      bool isLeft);

        // HUD overlay.
        void drawHUD(cv::Mat &img,
                     bool leftDetected, bool rightDetected,
                     double confLeft, double confRight);

        // CSV helpers and state.
        int frameId_ = 0;
        std::ofstream csv_;
        bool csvHeaderWritten_ = false;
        bool ensureCsvOpened();
        void writeCsvHeader();
        void writeCsvRow(int frameId, bool leftDetected, bool rightDetected, double confLeft, double confRight);
    };

}; // namespace ld
