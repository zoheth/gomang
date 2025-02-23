#pragma once

#include <memory>

#include "tensor.h"

namespace gomang
{
class IEngine
{
  public:
	virtual ~IEngine() = default;

	virtual IMemoryAllocator *getAllocator() = 0;

	// virtual std::unique_ptr<ITensor> createTensor(const TensorDesc &desc) = 0;

	virtual bool infer(
	    const std::vector<const void *> &inputs,
	    const std::vector<void *>       &outputs) = 0;

	virtual std::vector<TensorDesc> getInputInfo() const  = 0;
	virtual std::vector<TensorDesc> getOutputInfo() const = 0;

  protected:
	std::string  model_path_;
	unsigned int num_threads_;

	IEngine(const std::string &model_path, unsigned int num_threads) :
	    model_path_(model_path), num_threads_(num_threads)
	{}
};
}        // namespace gomang