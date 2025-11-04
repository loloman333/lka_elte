#include "lane_detector.h"
#include "utils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <map>
#include <cctype>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>

using namespace ld;

int main(int argc, char **argv) {
    std::string source;
    if (argc > 1) {
        source = argv[1];
    }

    cv::VideoCapture cap;
    if (source.empty()) {
        // try default camera
        cap.open(0);
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source. Provide a path or ensure a camera is available." << std::endl;
        return 1;
    }

    LaneDetector::Params params;
    LaneDetector detector(params);

    cv::namedWindow("Lane Detector", cv::WINDOW_NORMAL);
    cv::namedWindow("Debug", cv::WINDOW_NORMAL);

    bool showCrosshair = false; // toggle with 'o'

    // Prepare writers for each step
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || std::isnan(fps)) fps = 25.0;
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size frameSize(width > 0 ? width : 1280, height > 0 ? height : 720);
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1'); // H.264 if available; else fallback below

    auto safeWriter = [&](const std::string &path, bool isColor) {
        cv::VideoWriter w;
        if (!w.open(path, fourcc, fps, frameSize, isColor)) {
            // fallback to mp4v
            int f2 = cv::VideoWriter::fourcc('m','p','4','v');
            w.open(path, f2, fps, frameSize, isColor);
        }
        return w;
    };

    // Ensure output directory exists (portable without <filesystem>)
#ifdef _WIN32
    _mkdir("output");
#else
    if (mkdir("output", 0755) != 0 && errno != EEXIST) {
        std::perror("mkdir output");
    }
#endif

    cv::VideoWriter w_input = safeWriter("output/00_input.mp4", true);
    cv::VideoWriter w_out   = safeWriter("output/99_output.mp4", true);
    std::map<std::string, cv::VideoWriter> stepWriters; // key by step name

    auto sanitize = [](const std::string &name){
        std::string s; s.reserve(name.size());
        for (char c : name) s.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
        return s;
    };

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        PipelineDebug dbg;
        cv::Mat out = detector.processFrame(frame, &dbg);

        // Compose a debug mosaic from extensible steps (with labels)
        std::vector<cv::Mat> tiles;
        std::vector<std::string> labels;
        tiles.reserve(dbg.steps.size() + 1);
        labels.reserve(dbg.steps.size() + 1);
        tiles.push_back(showCrosshair ? ld::utils::withCrosshair(frame) : frame);
        labels.push_back("input");
        for (const auto &st : dbg.steps) {
            cv::Mat img = showCrosshair ? ld::utils::withCrosshair(st.image) : st.image;
            tiles.push_back(img);
            labels.push_back(st.name);
        }
        auto dbgMosaic = ld::utils::mosaicLabeled(tiles, labels, 3);
        ld::utils::putTextBox(out, "q: quit | space: pause | o: crosshair", {10, 25});

        cv::imshow("Lane Detector", out);
        if (!dbgMosaic.empty()) cv::imshow("Debug", dbgMosaic);

        // Write step videos (convert single-channel to 3-channel for color writers)
        if (!frame.empty()) {
            cv::Mat resized; cv::resize(frame, resized, frameSize);
            w_input.write(resized);
        }
        // Write per-step videos dynamically using dbg.steps
        for (size_t i = 0; i < dbg.steps.size(); ++i) {
            const auto &st = dbg.steps[i];
            cv::Mat img = st.image;
            if (img.empty()) continue;
            if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
            if (showCrosshair) img = ld::utils::withCrosshair(img);
            cv::Mat resized; cv::resize(img, resized, frameSize);

            std::string key = st.name;
            auto it = stepWriters.find(key);
            if (it == stepWriters.end()) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "output/%02zu_%s.mp4", i+1, sanitize(st.name).c_str());
                cv::VideoWriter writer = safeWriter(buf, true);
                stepWriters.emplace(key, std::move(writer));
                it = stepWriters.find(key);
            }
            if (it->second.isOpened()) it->second.write(resized);
        }
        if (!out.empty()) {
            cv::Mat resized; cv::resize(out, resized, frameSize);
            w_out.write(resized);
        }

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) break; // q or ESC
        if (key == ' ') {
            // pause until any key
            key = static_cast<char>(cv::waitKey(0));
            if (key == 'q' || key == 27) break;
        }
        if (key == 'o' || key == 'O') {
            showCrosshair = !showCrosshair;
        }
    }

    return 0;
}
