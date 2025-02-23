#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>

#include "core/engine.h"

namespace gomang
{
class Logger final : public nvinfer1::ILogger
{
  public:
	void log(Severity severity, const char *msg) noexcept override;
};

class TrtAllocator : public IMemoryAllocator
{
public:
	void* allocate(size_t size, MemoryType type) override;

	void deallocate(void* ptr, MemoryType type) override;
};

class TrtEngine : public IEngine
{
  public:
	explicit TrtEngine(const std::string &trt_model_path, unsigned int num_threads = 1);

	~TrtEngine() override;

	TrtEngine(const TrtEngine &)            = delete;
	TrtEngine(TrtEngine &&)                 = delete;
	TrtEngine &operator=(const TrtEngine &) = delete;
	TrtEngine &operator=(TrtEngine &&)      = delete;

	bool infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs) override;

	IMemoryAllocator *getAllocator() override;

	std::vector<TensorDesc> getInputInfo() const override;

	std::vector<TensorDesc> getOutputInfo() const override;

  protected:
	std::unique_ptr<nvinfer1::IRuntime>          trt_runtime_;
	std::unique_ptr<nvinfer1::ICudaEngine>       trt_engine_;
	std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
	Logger trt_logger_;

	cudaStream_t        stream_{};

	TrtAllocator allocator_;
	std::vector<std::shared_ptr<ITensor>> input_tensors_;
	std::vector<std::shared_ptr<ITensor>> output_tensors_;


  private:
	void initHandler();
	TensorDesc createTensorDesc(const char* tensor_name, const nvinfer1::Dims& dims, bool is_input);


};
}        // namespace gomang
