#include "trt_engine.h"

#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

TrtEngine::TrtEngine(const std::string &trt_model_path, unsigned int num_threads) :
    log_id(trt_model_path.data()),
    num_threads_(num_threads),
    trt_model_path_(trt_model_path.data())
{
	init_handler();
	print_debug_string();
}

TrtEngine::~TrtEngine()
{
	for (auto buffer : buffers)
	{
		cudaFree(buffer);
	}
	cudaStreamDestroy(stream_);
}

bool TrtEngine::benchmark(int num_warmup, int num_infer)
{
	if (!trt_context_)
	{
		std::cerr << "TensorRT context not initialized!" << std::endl;
		return false;
	}

	auto input_data = prepare_fake_input();

	std::cout << "Warmup..." << std::endl;
	for (int i = 0; i < num_warmup; ++i)
	{
		cudaMemcpyAsync(buffers[0], input_data.data(),
		                input_data.size() * sizeof(float),
		                cudaMemcpyHostToDevice, stream_);

		trt_context_->enqueueV3(stream_);
	}

	std::cout << "Benchmarking..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num_infer; ++i)
	{
		cudaMemcpyAsync(buffers[0], input_data.data(),
		                input_data.size() * sizeof(float),
		                cudaMemcpyHostToDevice, stream_);

		trt_context_->enqueueV3(stream_);
	}

	cudaStreamSynchronize(stream_);

	auto end      = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	float avg_time = duration.count() / 1000.0f / num_infer;
	std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
	std::cout << "FPS: " << 1000.0f / avg_time << std::endl;

	return true;
}

std::vector<float> TrtEngine::prepare_fake_input() const
{
	size_t input_size = 1;
	for (int dim : input_node_dims_)
	{
		input_size *= dim;
	}
	return std::vector<float>(input_size, 0.5f);
}

void TrtEngine::init_handler()
{
	// read engine file
	std::ifstream file(trt_model_path_, std::ios::binary);

	if (!file.good())
	{
		std::cerr << "Failed to read model file: " << trt_model_path_ << std::endl;
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
	buffers.resize(num_io_tensors);

	for (int i = 0; i < num_io_tensors; ++i)
	{
		auto           tensor_name = trt_engine_->getIOTensorName(i);
		nvinfer1::Dims tensor_dims = trt_engine_->getTensorShape(tensor_name);

		// input
		if (i == 0)
		{
			size_t tensor_size = 1;
			for (int j = 0; j < tensor_dims.nbDims; ++j)
			{
				tensor_size *= tensor_dims.d[j];
				input_node_dims_.push_back(tensor_dims.d[j]);
			}
			cudaMalloc(&buffers[i], tensor_size * sizeof(float));
			trt_context_->setTensorAddress(tensor_name, buffers[i]);
			continue;
		}

		// output
		size_t tensor_size = 1;

		std::vector<int64_t> output_node;
		for (int j = 0; j < tensor_dims.nbDims; ++j)
		{
			output_node.push_back(tensor_dims.d[j]);
			tensor_size *= tensor_dims.d[j];
		}
		output_node_dims_.push_back(output_node);

		cudaMalloc(&buffers[i], tensor_size * sizeof(float));
		trt_context_->setTensorAddress(tensor_name, buffers[i]);
		output_tensor_size_++;
	}
}

void TrtEngine::print_debug_string()
{
	std::cout << "TensorRT model loaded from: " << trt_model_path_ << std::endl;
	std::cout << "Input tensor size: " << input_tensor_size_ << std::endl;
	std::cout << "Output tensor size: " << output_tensor_size_ << std::endl;
}
