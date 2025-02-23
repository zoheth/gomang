#include "benchmark.h"
#include <chrono>
#ifdef ENABLE_TENSORRT
#	include "backends/trt/trt_engine.h"
#endif

#ifdef ENABLE_MNN
#	include "backends/mnn/mnn_engine.h"
#endif

#ifdef ENABLE_NCNN
#include "backends/ncnn/ncnn_engine.h"
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

#ifdef ENABLE_NCNN
	{
		gomang::TensorDesc input_desc;
		input_desc.shape = {1,3,640,640};
		input_desc.data_type = gomang::DataType::kFLOAT32;
		input_desc.layout = gomang::MemoryLayout::kNCHW;
		input_desc.mem_type = gomang::MemoryType::kCPU_PINNED;

		auto engine = std::make_shared<gomang::NcnnEngine>("models/yolov5s.ncnn", input_desc);
		// auto engine = std::make_shared<gomang::NcnnEngine>("models/nanodet_m-opt");
		auto bench = gomang::Benchmark(engine);
		bench.run(2, 5);
	}
#endif

	return 0;
}
