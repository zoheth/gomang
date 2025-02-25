#include "benchmark.h"
#include <chrono>
#ifdef ENABLE_TENSORRT
#	include "backends/trt/trt_engine.h"
#endif

#ifdef ENABLE_MNN
#	include "backends/mnn/mnn_engine.h"
#endif

#ifdef ENABLE_NCNN
#	include "backends/ncnn/ncnn_engine.h"
#endif

int main()
{
	std::string model_names[] = {
	    "yolov5s",
	    "D_dncnn_color_blind",
	    "D_ircnn_color",
	    "ED_table5_pidinet_tiny_l",
	    "ED_table5_pidinet_tiny",
	    "SR_edsr",
	    "SR_msrresnet_x4_psnr",
	    "ST_mosaic_d"};
	std::string model_name = model_names[3];

	std::cout << "Benchmarking " << model_name << " model" << std::endl;

#ifdef ENABLE_TENSORRT
	{
		auto engine = std::make_shared<gomang::TrtEngine>("models/trt/" + model_name + ".engine");
		auto bench  = gomang::Benchmark(engine);
		bench.run();
	}
#endif

#ifdef ENABLE_MNN
	{
		auto engine = std::make_shared<gomang::MnnEngine>("models/mnn/" + model_name + ".mnn");
		auto bench  = gomang::Benchmark(engine);
		bench.run();
	}
#endif

#ifdef ENABLE_NCNN
	{
		gomang::TensorDesc input_desc;
		input_desc.shape     = {1, 3, 640, 640};
		input_desc.data_type = gomang::DataType::kFLOAT32;
		input_desc.layout    = gomang::MemoryLayout::kNCHW;
		input_desc.mem_type  = gomang::MemoryType::kCPU_PINNED;

		auto engine = std::make_shared<gomang::NcnnEngine>("models/ncnn/" + model_name + ".ncnn", input_desc);
		// auto engine = std::make_shared<gomang::NcnnEngine>("models/nanodet_m-opt");
		auto bench = gomang::Benchmark(engine);
		bench.run(0, 1);
	}
#endif

	return 0;
}
