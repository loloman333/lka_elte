# Simple Makefile for Lane Detector (C++17 + OpenCV)

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -MMD -MP -Iinclude
LDFLAGS :=

# Try to find OpenCV via pkg-config first
OPENCV_PKGCFG := $(shell pkg-config --exists opencv4 && echo opencv4 || (pkg-config --exists opencv && echo opencv || echo none))

ifeq ($(OPENCV_PKGCFG),none)
	# Fallback: common include/lib locations
	OPENCV_INC_FALLBACK := $(shell if [ -d /usr/local/include/opencv4 ]; then echo -I/usr/local/include/opencv4; elif [ -d /usr/include/opencv4 ]; then echo -I/usr/include/opencv4; else echo; fi)
	OPENCV_LIBDIR_FALLBACK := $(shell if [ -d /usr/local/lib ]; then echo -L/usr/local/lib; elif [ -d /usr/lib/x86_64-linux-gnu ]; then echo -L/usr/lib/x86_64-linux-gnu; elif [ -d /usr/lib ]; then echo -L/usr/lib; else echo; fi)
	OPENCV_WORLD := $(shell ls /usr/local/lib/libopencv_world* >/dev/null 2>&1 || ls /usr/lib/libopencv_world* >/dev/null 2>&1 || ls /usr/lib/x86_64-linux-gnu/libopencv_world* >/dev/null 2>&1 && echo -lopencv_world || echo)
	ifeq ($(OPENCV_WORLD),)
		OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs
	else
		OPENCV_LIBS := $(OPENCV_WORLD)
	endif
	CXXFLAGS += $(OPENCV_INC_FALLBACK)
	LDFLAGS += $(OPENCV_LIBDIR_FALLBACK) $(OPENCV_LIBS) -Wl,-rpath,/usr/local/lib
else
	CXXFLAGS += $(shell pkg-config --cflags $(OPENCV_PKGCFG))
	LDFLAGS += $(shell pkg-config --libs $(OPENCV_PKGCFG))
endif

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
TARGET := lane_detector

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)

.PHONY: all run clean format

all: $(BIN_DIR)/$(TARGET)

$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

run: $(BIN_DIR)/$(TARGET)
	./$(BIN_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

format:
	@command -v clang-format >/dev/null 2>&1 && \
	  clang-format -i $(SOURCES) $(wildcard $(INC_DIR)/*.h) || \
	  echo "clang-format not found, skipping"

# Include auto-generated dependencies
-include $(DEPS)
