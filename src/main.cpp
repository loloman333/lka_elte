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
    bool enableDebug = true;   // --no-debug or --fast disables this
    int stride = 1;            // --stride=N or implied by --fast
    double scale = 1.0;        // --scale=F or implied by --fast

    // Parse args: first non-flag is source; flags can appear in any order
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-debug") {
            enableDebug = false;
        } else if (arg == "--fast") {
            enableDebug = false;
            stride = 2;
            scale = 0.5;
        } else if (arg.rfind("--scale=", 0) == 0) {
            try { scale = std::max(0.05, std::stod(arg.substr(8))); } catch (...) {}
        } else if (arg.rfind("--stride=", 0) == 0) {
            try { stride = std::max(1, std::stoi(arg.substr(9))); } catch (...) {}
        } else if (source.empty()) {
            source = arg;
        }
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

    // Prepare writers for each step
    double srcFps = cap.get(cv::CAP_PROP_FPS);
    if (srcFps <= 0 || std::isnan(srcFps)) srcFps = 25.0;
    double writerFps = srcFps / std::max(1, stride);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size baseSize(width > 0 ? width : 1280, height > 0 ? height : 720);
    cv::Size frameSize(
        std::max(1, static_cast<int>(baseSize.width * scale + 0.5)),
        std::max(1, static_cast<int>(baseSize.height * scale + 0.5))
    );
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1'); // H.264 if available; else fallback below

    auto safeWriter = [&](const std::string &path, bool isColor) {
        cv::VideoWriter w;
        if (!w.open(path, fourcc, writerFps, frameSize, isColor)) {
            // fallback to mp4v
            int f2 = cv::VideoWriter::fourcc('m','p','4','v');
            w.open(path, f2, writerFps, frameSize, isColor);
        }
        return w;
    };

    // Create UI windows with sensible default sizes (after frameSize is known)
    {
        int minW = 960, minH = 540;
        cv::namedWindow("Lane Detector", cv::WINDOW_NORMAL);
        cv::resizeWindow("Lane Detector",
                         std::max(frameSize.width,  minW),
                         std::max(frameSize.height, minH));

        if (enableDebug) {
            cv::namedWindow("Debug", cv::WINDOW_NORMAL);
            // Debug mosaic tends to be larger; give it more room by default.
            cv::resizeWindow("Debug",
                             std::max(frameSize.width  * 2, minW),
                             std::max(frameSize.height * 2, minH));
            // Optional: place Debug to the right of the main window
            // cv::moveWindow("Debug", std::max(frameSize.width, minW) + 40, 40);
        }
    }

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

    long long frameIdx = 0;
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        ++frameIdx;

        // Process every 'stride'-th frame to save time
        if (stride > 1 && ((frameIdx - 1) % stride) != 0) continue;

        // Downscale for faster processing if requested
        cv::Mat proc = frame;
        if (scale != 1.0) cv::resize(frame, proc, frameSize);

        PipelineDebug dbg;
        PipelineDebug* dbgPtr = enableDebug ? &dbg : nullptr;
        cv::Mat out = detector.processFrame(proc, dbgPtr);

        // Debug UI and per-step mosaic only when enabled
        if (enableDebug) {
            std::vector<cv::Mat> tiles;
            std::vector<std::string> labels;
            tiles.reserve(dbg.steps.size() + 1);
            labels.reserve(dbg.steps.size() + 1);
            tiles.push_back(proc); // removed crosshair overlay
            labels.push_back("input");
            for (const auto &st : dbg.steps) {
                cv::Mat img = st.image;
                tiles.push_back(img);
                labels.push_back(st.name);
            }
            auto dbgMosaic = ld::utils::mosaicLabeled(tiles, labels, 3);
            if (!dbgMosaic.empty()) cv::imshow("Debug", dbgMosaic);

            // Write per-step videos (convert single-channel to 3-channel for color writers)
            for (size_t i = 0; i < dbg.steps.size(); ++i) {
                const auto &st = dbg.steps[i];
                cv::Mat img = st.image;
                if (img.empty()) continue;
                if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                // removed crosshair overlay
                cv::Mat resized;
                if (img.size() != frameSize) cv::resize(img, resized, frameSize); else resized = img;

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
        }

        // Display and output writing on scaled frames to reduce work
        if (!out.empty()) cv::imshow("Lane Detector", out);

        if (w_input.isOpened()) {
            // 'proc' is already at frameSize
            w_input.write(proc);
        }
        if (!out.empty() && w_out.isOpened()) {
            // 'out' is already at frameSize
            w_out.write(out);
        }

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 27) break; // q or ESC
        if (key == ' ') {
            // pause until any key
            key = static_cast<char>(cv::waitKey(0));
            if (key == 'q' || key == 27) break;
        }
    }

    return 0;
}
