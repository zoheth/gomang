#pragma once

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <vector>

class Logger final : public nvinfer1::ILogger
{
  public:
	void log(Severity severity, const char *msg) noexcept override
	{
		if (severity != Severity::kINFO)
		{
			std::cout << "TensorRT: " << msg << std::endl;
		}
	}
};

class TrtEngine
{
  public:
	explicit TrtEngine(const std::string &trt_model_path, unsigned int num_threads = 1);

	virtual ~TrtEngine();

	TrtEngine(const TrtEngine &)            = delete;
	TrtEngine(TrtEngine &&)                 = delete;
	TrtEngine &operator=(const TrtEngine &) = delete;
	TrtEngine &operator=(TrtEngine &&)      = delete;

	bool benchmark(int num_warmup = 10, int num_infer = 100);

  protected:
	std::vector<float> prepare_fake_input() const;

  protected:
	std::unique_ptr<nvinfer1::IRuntime>          trt_runtime_;
	std::unique_ptr<nvinfer1::ICudaEngine>       trt_engine_;
	std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;

	Logger trt_logger_;

	std::vector<void *> buffers;
	cudaStream_t        stream_{};

	std::vector<int64_t>              input_node_dims_;
	std::vector<std::vector<int64_t>> output_node_dims_;
	std::size_t                       input_tensor_size_ = 1;
	std::size_t                       output_tensor_size_{0};

	const char        *trt_model_path_{nullptr};
	const char        *log_id{nullptr};
	const unsigned int num_threads_;

  private:
	void init_handler();
	void print_debug_string();
};
