#include "benchmark.h"
#include <chrono>
#ifdef ENABLE_TENSORRT
#	include "backends/trt/trt_engine.h"
#endif

#ifdef ENABLE_MNN
#	include "backends/mnn/mnn_engine.h"
#endif

int main()
{
#ifdef ENABLE_TENSORRT
	auto trt_engine = std::make_shared<gomang::TrtEngine>("models/yolov5s_fp32.engine");
	auto trt_bench  = gomang::Benchmark(trt_engine);
	trt_bench.run();
#endif

#ifdef ENABLE_MNN
	MnnEngine mnn_engine("models/yolov5s.mnn");
	mnn_engine.benchmark(2, 5);
#endif

	return 0;
}
