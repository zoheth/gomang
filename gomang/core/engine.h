#pragma once

#include <iostream>
#include <memory>
#include <utility>

#include "tensor.h"

namespace gomang
{
class IEngine
{
  public:
	virtual ~IEngine() = default;

	IEngine(const IEngine &)            = delete;
	IEngine(IEngine &&)                 = delete;
	IEngine &operator=(const IEngine &) = delete;
	IEngine &operator=(IEngine &&)      = delete;

	// virtual std::unique_ptr<ITensor> createTensor(const TensorDesc &desc) = 0;

	virtual bool infer(
	    const std::vector<const void *> &inputs,
	    const std::vector<void *>       &outputs) = 0;

	[[nodiscard]] virtual std::vector<TensorDesc> getInputInfo() const  = 0;
	[[nodiscard]] virtual std::vector<TensorDesc> getOutputInfo() const = 0;

	[[nodiscard]] const std::string &getName() const;

	void printTensorInfo() const;

  protected:
	std::string  model_path_;
	unsigned int num_threads_;

	std::string name_{};

	IEngine(std::string model_path, unsigned int num_threads, std::string name);
};
}        // namespace gomang
