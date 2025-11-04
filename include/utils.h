#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace ld::utils {

// Put semi-transparent text box on image
void putTextBox(cv::Mat &img, const std::string &text, const cv::Point &org,
                double fontScale = 0.6, int thickness = 1,
                const cv::Scalar &textColor = {255, 255, 255},
                const cv::Scalar &bgColor = {0, 0, 0}, double alpha = 0.4);

// Create a simple side-by-side mosaic for debugging
cv::Mat mosaic(const std::vector<cv::Mat> &images, int cols = 2, int cellW = 480, int cellH = 270);

// Create a mosaic with labels drawn at top-left of each tile
cv::Mat mosaicLabeled(const std::vector<cv::Mat> &images, const std::vector<std::string> &labels,
                      int cols = 2, int cellW = 480, int cellH = 270);

// Return a copy of the image with a crosshair (vertical + horizontal) overlay.
// Ensures output is BGR. Blue by default.
cv::Mat withCrosshair(const cv::Mat &img, const cv::Scalar &color = {255, 0, 0}, int thickness = 1);

} // namespace ld::utils
