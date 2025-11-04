#include "utils.h"

#include <opencv2/imgproc.hpp>

namespace ld::utils {

void putTextBox(cv::Mat &img, const std::string &text, const cv::Point &org,
                double fontScale, int thickness, const cv::Scalar &textColor,
                const cv::Scalar &bgColor, double alpha) {
    int baseline = 0;
    auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    cv::Rect box(org.x - 4, org.y - size.height - 6, size.width + 8, size.height + baseline + 8);
    box &= cv::Rect(0, 0, img.cols, img.rows);
    if (box.width > 0 && box.height > 0) {
        cv::Mat roi = img(box);
        cv::Mat overlay; roi.copyTo(overlay);
        cv::rectangle(overlay, cv::Rect(0, 0, roi.cols, roi.rows), bgColor, cv::FILLED);
        cv::addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi);
    }
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness, cv::LINE_AA);
}

cv::Mat mosaic(const std::vector<cv::Mat> &images, int cols, int cellW, int cellH) {
    if (images.empty()) return cv::Mat();
    int rows = static_cast<int>((images.size() + cols - 1) / cols);
    cv::Mat canvas(rows * cellH, cols * cellW, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < images.size(); ++i) {
        int r = static_cast<int>(i / cols);
        int c = static_cast<int>(i % cols);
        cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
        cv::Mat dst = canvas(cell);
        cv::Mat src;
        if (images[i].channels() == 1)
            cv::cvtColor(images[i], src, cv::COLOR_GRAY2BGR);
        else
            src = images[i];
        cv::Mat resized;
        cv::resize(src, resized, dst.size(), 0, 0, cv::INTER_AREA);
        resized.copyTo(dst);
    }
    return canvas;
}

cv::Mat mosaicLabeled(const std::vector<cv::Mat> &images, const std::vector<std::string> &labels,
                      int cols, int cellW, int cellH) {
    if (images.empty()) return cv::Mat();
    int rows = static_cast<int>((images.size() + cols - 1) / cols);
    cv::Mat canvas(rows * cellH, cols * cellW, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < images.size(); ++i) {
        int r = static_cast<int>(i / cols);
        int c = static_cast<int>(i % cols);
        cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
        cv::Mat dst = canvas(cell);
        cv::Mat src;
        if (images[i].channels() == 1)
            cv::cvtColor(images[i], src, cv::COLOR_GRAY2BGR);
        else
            src = images[i];
        cv::Mat resized;
        cv::resize(src, resized, dst.size(), 0, 0, cv::INTER_AREA);
        resized.copyTo(dst);

        if (i < labels.size()) {
            std::string text = labels[i];
            putTextBox(canvas, text, cv::Point(cell.x + 8, cell.y + 24), 0.6, 1,
                       cv::Scalar(255,255,255), cv::Scalar(0,0,0), 0.35);
        }
    }
    return canvas;
}

cv::Mat withCrosshair(const cv::Mat &img, const cv::Scalar &color, int thickness) {
    if (img.empty()) return img;
    cv::Mat out;
    if (img.channels() == 1)
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else
        out = img.clone();

    int cx = out.cols / 2;
    int cy = out.rows / 2;
    cv::line(out, {cx, 0}, {cx, out.rows - 1}, color, thickness, cv::LINE_AA);
    cv::line(out, {0, cy}, {out.cols - 1, cy}, color, thickness, cv::LINE_AA);
    return out;
}

} // namespace ld::utils
