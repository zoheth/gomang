#include "benchmark.h"
#include <chrono>

#ifdef ENABLE_TENSORRT
	#include "backends/trt/trt_engine.h"
#endif

int main()
{
	std::string model_name = "nvh";

	std::cout << "Benchmarking " << model_name << " model" << std::endl;

#ifdef ENABLE_TENSORRT
{
	auto engine = std::make_shared<gomang::TrtEngine>("/home/x/dev/gomang/models/trt/nbvh.engine");
	auto bench  = gomang::Benchmark(engine);
	bench.run(0, 1);
}
#endif

	return 0;
}
