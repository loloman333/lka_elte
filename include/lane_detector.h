#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ld
{

    struct DebugStep
    {
        std::string name;
        cv::Mat image;
    };

    struct PipelineDebug
    {
        // Extensible ordered list of debug step images
        std::vector<DebugStep> steps;
    };

    class LaneDetector
    {
    public:
        // Parameters struct for easy tuning
        struct Params
        {
            int cannyLow = 50;
            int cannyHigh = 120;
            int gaussKernel = 5;
            double houghRho = 1.0;
            double houghTheta = CV_PI / 180.0;
            int houghThreshold = 20;
            double houghMinLineLength = 100.0;
            double houghMaxLineGap = 100.0;
            // ROI (normalized): use rectangle covering lower 2/3 of the frame
            // top at 1/3 (0.333..), bottom at 1.0
            double roiTopY = 1.0 / 3.0;
            double roiBottomY = 1.0;
        };

        LaneDetector() = default;
        explicit LaneDetector(const Params &p) : params_(p) {}

        // Process one BGR frame and return an annotated BGR frame
        cv::Mat processFrame(const cv::Mat &bgrFrame, PipelineDebug *dbg = nullptr);

    private:
        Params params_{};

        cv::Mat convertToHLS(const cv::Mat &bgr);
        cv::Mat detectEdges(const cv::Mat &bgrOrGray);
        // Angle from vertical at the center-top cut; can exceed 90°. 0 keeps rectangular ROI.
        cv::Mat roiMask(const cv::Size &size, double ratio = 0.66, double angleDeg = 0.0);
        std::vector<cv::Vec4i> detectLines(const cv::Mat &edges);
        std::pair<cv::Vec4i, cv::Vec4i>
        pickFirstLinesByBottom(const std::vector<cv::Vec4i> &lines, int imgWidth, int imgHeight, int minLen = 30, double minAbsSlope = 0.15);
        // New: raycast-based seed picker (left/right lists)
        std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
        pickFirstLines(const std::vector<cv::Vec4i> &lines, int imgWidth, int imgHeight, int numRays = 3);

        // New: RANSAC-like seed filter to remove outliers
        std::vector<cv::Vec4i>
        ransacFilterSeedLines(const std::vector<cv::Vec4i> &seeds, int imgHeight, int iterations = 50, double inlierThreshPx = 10.0);

        // New pipeline helpers
        bool fitQuadraticXY(const std::vector<cv::Point> &points, cv::Vec3d &coeffs); // x = a*y^2 + b*y + c
        void drawQuadratic(cv::Mat &img, const cv::Vec3d &abc, int topY, int bottomY, const cv::Scalar &color, int thickness = 3);
        // Fit x = a*y^2 + b*y + c on a set of points (Mat of cv::Point-like entries) and return [a,b,c]^T
        cv::Mat fitPolynomial(const cv::Mat &points);

        // Group Hough lines that agree with seed left/right lines
        std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
        findAgreeingLines(const std::vector<cv::Vec4i> &houghLines,
                          const std::vector<cv::Vec4i> &seedLeft,
                          const std::vector<cv::Vec4i> &seedRight,
                          double slopeTol = 0.2,
                          double maxEndpointDist = 10.0);

        // Helper used by findAgreeingLines for one side
        std::vector<cv::Vec4i>
        findAgreeingLinesForSide(const std::vector<cv::Vec4i> &houghLines,
                                 const std::vector<cv::Vec4i> &seedLines,
                                 double slopeTol,
                                 double maxEndpointDist);

        // Drawing helpers
        void drawLine(cv::Mat &img, const cv::Vec4i &line, const cv::Scalar &color, int thickness = 1);
        void drawLines(cv::Mat &img, const std::vector<cv::Vec4i> &lines, const cv::Scalar &color, int thickness = 1);

        // New: builds a lane mask from edges and selected lines (currently uses leftLines only)
        cv::Mat buildLaneMask(const cv::Mat &maskedEdges,
                              const std::vector<cv::Vec4i> &leftLines,
                              const std::vector<cv::Vec4i> &rightLines);
    };

}; // namespace ld
