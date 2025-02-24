#include "trt_engine.h"

#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

namespace gomang
{
void Logger::log(Severity severity, const char *msg) noexcept
{
	if (severity != Severity::kINFO)
	{
		std::cout << "TensorRT: " << msg << std::endl;
	}
}
void *TrtAllocator::allocate(size_t size, MemoryType type)
{
	void *ptr = nullptr;
	if (type == MemoryType::kGPU)
	{
		cudaMalloc(&ptr, size);
	}
	else if (type == MemoryType::kCPU_PINNED)
	{
		cudaMallocHost(&ptr, size);
	}
	else
	{
		std::cerr << "Unsupported memory type" << std::endl;
	}
	return ptr;
}
void TrtAllocator::deallocate(void *ptr, MemoryType type)
{
	if (type == MemoryType::kGPU)
	{
		cudaFree(ptr);
	}
	else if (type == MemoryType::kCPU_PINNED)
	{
		cudaFreeHost(ptr);
	}
	else
	{
		std::free(ptr);
	}
}

TrtEngine::TrtEngine(const std::string &model_path, unsigned int num_threads) :
    IEngine(model_path, num_threads, "TensorRT")
{
	initHandler();
}

TrtEngine::~TrtEngine()
{
	input_tensors_.clear();
	output_tensors_.clear();
	cudaStreamDestroy(stream_);
}
bool TrtEngine::infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs)
{
	if (!trt_context_ || inputs.size() != input_tensors_.size() ||
	    outputs.size() != output_tensors_.size())
	{
		return false;
	}

	for (size_t i = 0; i < inputs.size(); ++i)
	{
		size_t size = input_tensors_[i]->size();
		cudaMemcpyAsync(input_tensors_[i]->data(), inputs[i], size,
		                cudaMemcpyHostToDevice, stream_);
	}
	cudaStreamSynchronize(stream_);

	if (!trt_context_->enqueueV3(stream_))
	{
		return false;
	}

	for (size_t i = 0; i < outputs.size(); ++i)
	{
		cudaStreamSynchronize(stream_);
		size_t size = output_tensors_[i]->size();
		cudaMemcpyAsync(outputs[i], output_tensors_[i]->data(), size,
		                cudaMemcpyDeviceToHost, stream_);
	}

	cudaStreamSynchronize(stream_);
	return true;
}

std::vector<TensorDesc> TrtEngine::getInputInfo() const
{
	std::vector<TensorDesc> res;
	res.reserve(input_tensors_.size());
	for (const auto &tensor : input_tensors_)
	{
		res.push_back(tensor->desc());
	}
	return res;
}
std::vector<TensorDesc> TrtEngine::getOutputInfo() const
{
	std::vector<TensorDesc> res;
	res.reserve(output_tensors_.size());
	for (const auto &tensor : output_tensors_)
	{
		res.push_back(tensor->desc());
	}
	return res;
}

TensorDesc TrtEngine::createTensorDesc(const char *tensor_name, const nvinfer1::Dims &dims, bool is_input)
{
	TensorDesc desc;
	desc.name = tensor_name;
	desc.shape.resize(dims.nbDims);
	for (int i = 0; i < dims.nbDims; i++)
	{
		desc.shape[i] = dims.d[i];
	}

	desc.layout    = MemoryLayout::kNCHW;
	desc.data_type = DataType::kFLOAT32;
	desc.mem_type  = MemoryType::kGPU;

	return desc;
}

void TrtEngine::initHandler()
{
	// read engine file
	std::ifstream file(model_path_, std::ios::binary);

	if (!file.good())
	{
		std::cerr << "Failed to read model file: " << model_path_ << std::endl;
		return;
	}
	file.seekg(0, std::ifstream::end);
	size_t model_size = file.tellg();
	file.seekg(0, std::ifstream::beg);
	std::vector<char> model_data(model_size);
	file.read(model_data.data(), model_size);
	file.close();

	trt_runtime_.reset(nvinfer1::createInferRuntime(trt_logger_));
	// engine deserialize
	trt_engine_.reset(trt_runtime_->deserializeCudaEngine(model_data.data(), model_size));
	if (!trt_engine_)
	{
		std::cerr << "Failed to deserialize the TensorRT engine." << std::endl;
		return;
	}
	trt_context_.reset(trt_engine_->createExecutionContext());
	if (!trt_context_)
	{
		std::cerr << "Failed to create execution context." << std::endl;
		return;
	}
	cudaStreamCreate(&stream_);

	// make the flexible one input and multi output
	int num_io_tensors = trt_engine_->getNbIOTensors();        // get the input and output's num

	for (int i = 0; i < num_io_tensors; ++i)
	{
		auto           tensor_name = trt_engine_->getIOTensorName(i);
		nvinfer1::Dims tensor_dims = trt_engine_->getTensorShape(tensor_name);

		bool       is_input = (i == 0);
		TensorDesc desc     = createTensorDesc(tensor_name, tensor_dims, is_input);
		auto       tensor   = std::make_shared<Tensor>(desc, &allocator_);
		trt_context_->setTensorAddress(tensor_name, tensor->data());

		if (is_input)
		{
			input_tensors_.push_back(tensor);
		}
		else
		{
			output_tensors_.push_back(tensor);
		}
	}
}

}        // namespace gomang
