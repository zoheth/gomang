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

	void run(int num_warmup = 10, int num_infer = 100) const;

  private:
	std::shared_ptr<IEngine> engine_;
};
}        // namespace gomang