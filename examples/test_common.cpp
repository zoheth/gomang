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
	{
		auto engine = std::make_shared<gomang::TrtEngine>("models/yolov5s_fp32.engine");
		auto bench  = gomang::Benchmark(engine);
		bench.run();
	}
#endif

#ifdef ENABLE_MNN
	{
		auto engine = std::make_shared<gomang::MnnEngine>("models/yolov5s.mnn");
		auto bench  = gomang::Benchmark(engine);
		bench.run(2, 5);
	}
#endif

	return 0;
}
