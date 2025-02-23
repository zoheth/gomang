#include <chrono>

#include "backends/trt/trt_engine.h"

int main() {
    TrtEngine trt_yolox("models/yolov5s_fp32.engine");
    trt_yolox.benchmark();
    return 0;
}
