#include "benchmark.h"

namespace gomang
{

namespace
{
void printSimpleOutputCheck(const std::vector<std::vector<float>>& output_buffers, int sample_count = 5) {
	std::cout << "=== Output Check ===" << std::endl;

	for (size_t i = 0; i < output_buffers.size(); ++i) {
		const auto& buffer = output_buffers[i];
		if (buffer.empty()) {
			std::cout << "Buffer " << i << ": [Empty]" << std::endl;
			continue;
		}

		// 找出最大值和最小值
		float min_val = buffer[0];
		float max_val = buffer[0];
		for (const auto& val : buffer) {
			if (val < min_val) min_val = val;
			if (val > max_val) max_val = val;
		}

		// 输出基本信息
		std::cout << "Buffer " << i << " (size: " << buffer.size()
				  << ", min: " << min_val << ", max: " << max_val << ")" << std::endl;

		// 输出开头的几个值
		std::cout << "  Front values: ";
		int count = std::min(sample_count, static_cast<int>(buffer.size()));
		for (int j = 0; j < count; ++j) {
			std::cout << buffer[j];
			if (j < count - 1) std::cout << ", ";
		}
		std::cout << std::endl;
	}

	std::cout << "===================" << std::endl;
}
}

void Benchmark::run(int num_warmup, int num_infer) const
{
	std::cout << std::endl
	          << "[[" << engine_->getName() << "]]:" << std::endl;
	engine_->printTensorInfo();

	auto               input_info = engine_->getInputInfo();
	std::vector<float> input_data(input_info[0].getElementsCount(), 0.f);

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

	printSimpleOutputCheck(output_buffers);

	float avg_time = duration.count() / 1000.0f / num_infer;
	std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
	std::cout << "FPS: " << 1000.0f / avg_time << std::endl << std::endl;
}
}        // namespace gomang