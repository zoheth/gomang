#include "benchmark.h"

namespace gomang
{

void Benchmark::run(int num_warmup, int num_infer) const
{
	std::cout << std::endl
	          << "[[" << engine_->getName() << "]]:" << std::endl;
	engine_->printTensorInfo();

	auto               input_info = engine_->getInputInfo();
	std::vector<float> input_data(input_info[0].getElementsCount());

	std::vector<void *>             outputs;
	std::vector<std::vector<float>> output_buffers;
	for (const auto &desc : engine_->getOutputInfo())
	{
		output_buffers.emplace_back(desc.getElementsCount());
		outputs.push_back(output_buffers.back().data());
	}

	std::vector<const void *> inputs = {input_data.data()};

	std::cout << "Warmup..." << std::endl;
	for (int i = 0; i < num_warmup; ++i)
	{
		engine_->infer(inputs, outputs);
	}

	std::cout << "Benchmarking..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num_infer; ++i)
	{
		engine_->infer(inputs, outputs);
	}

	auto end      = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	float avg_time = duration.count() / 1000.0f / num_infer;
	std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
	std::cout << "FPS: " << 1000.0f / avg_time << std::endl << std::endl;
}
}        // namespace gomang