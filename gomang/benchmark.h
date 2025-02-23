#pragma once

#include <utility>

#include "core/engine.h"

#include <chrono>
#include <iostream>
namespace gomang
{
class Benchmark
{
  public:
	explicit Benchmark(std::shared_ptr<IEngine> engine) :
	    engine_(std::move(engine))
	{}

	struct BenchmarkResult
	{
		float avg_time_ms;
		float fps;
		int   num_iterations;
	};

	BenchmarkResult run(int num_warmup = 10, int num_infer = 100)
	{
		auto input_info = engine_->getInputInfo();

		std::vector<float> input_data(input_info[0].getElementsCount());

		std::vector<void *>       outputs;
		std::vector<std::vector<float>> output_buffers;
		for (const auto & desc : engine_->getOutputInfo())
		{
			output_buffers.emplace_back(desc.getElementsCount());
			outputs.push_back(output_buffers.back().data());
		}


		std::vector<const void *> inputs  = {input_data.data()};

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

		BenchmarkResult result;
		result.avg_time_ms    = duration.count() / 1000.0f / num_infer;
		result.fps            = 1000.0f / result.avg_time_ms;
		result.num_iterations = num_infer;

		float avg_time = duration.count() / 1000.0f / num_infer;
		std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
		std::cout << "FPS: " << 1000.0f / avg_time << std::endl;

		return result;
	}

  private:
	std::shared_ptr<IEngine> engine_;
};
}        // namespace gomang