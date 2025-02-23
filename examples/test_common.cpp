#include <chrono>
#ifdef ENABLE_TENSORRT
#include "backends/trt/trt_engine.h"
#endif

#ifdef ENABLE_MNN
#include "backends/mnn/mnn_engine.h"
#endif

int main() {
#ifdef ENABLE_TENSORRT
    TrtEngine trt_yolox("models/yolov5s_fp32.engine");
    trt_yolox.benchmark();
#endif

#ifdef ENABLE_MNN
	MnnEngine mnn_engine("models/yolov5s.mnn");
	mnn_engine.benchmark();
#endif

    return 0;
}
