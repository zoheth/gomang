#include "mnn_engine.h"

#include <chrono>
#include <iostream>

MnnEngine::MnnEngine(const std::string &mnn_path, unsigned int num_threads) :
    log_id_(mnn_path.data()), mnn_path_(mnn_path.data()), num_threads_(num_threads)
{
	init_handler();
}

MnnEngine::~MnnEngine()
{
	mnn_interpreter_->releaseModel();
	if (mnn_session_)
	{
		mnn_interpreter_->releaseSession(mnn_session_);
	}
}

bool MnnEngine::benchmark(int num_warmup, int num_infer)
{
	if (!mnn_interpreter_ || !mnn_session_)
	{
		std::cerr << "MNN interpreter or session not initialized!" << std::endl;
		return false;
	}

	std::cout << "Warmup..." << std::endl;
	for (int i = 0; i < num_warmup; ++i)
	{
		prepare_fake_input();
		mnn_interpreter_->runSession(mnn_session_);
	}

	std::cout << "Benchmarking..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num_infer; ++i)
	{
		prepare_fake_input();
		mnn_interpreter_->runSession(mnn_session_);
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	float avg_time = duration.count() / 1000.0f / num_infer;
	std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
	std::cout << "FPS: " << 1000.0f / avg_time << std::endl;

	return true;
}
void MnnEngine::init_handler()
{
	mnn_interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path_));

	schedule_config_.numThread = static_cast<int>(num_threads_);
	MNN::BackendConfig backend_config;
	backend_config.precision       = MNN::BackendConfig::Precision_High;
	schedule_config_.backendConfig = &backend_config;

	mnn_session_ = mnn_interpreter_->createSession(schedule_config_);

	input_tensor_ = mnn_interpreter_->getSessionInput(mnn_session_, nullptr);

	input_batch_    = input_tensor_->batch();
	input_channel_  = input_tensor_->channel();
	input_height_   = input_tensor_->height();
	input_width_    = input_tensor_->width();
	dimension_type_ = input_tensor_->getDimensionType();

	if (dimension_type_ == MNN::Tensor::CAFFE)
	{
		// NCHW
		mnn_interpreter_->resizeTensor(input_tensor_, {input_batch_, input_channel_, input_height_, input_width_});
		mnn_interpreter_->resizeSession(mnn_session_);
	}
	else if (dimension_type_ == MNN::Tensor::TENSORFLOW)
	{
		// NHWC
		mnn_interpreter_->resizeTensor(input_tensor_, {input_batch_, input_height_, input_width_, input_channel_});
		mnn_interpreter_->resizeSession(mnn_session_);
	}
	else if (dimension_type_ == MNN::Tensor::CAFFE_C4)
	{
		// do nothing
	}

	num_outputs_ = static_cast<int>(mnn_interpreter_->getSessionOutputAll(mnn_session_).size());

	print_debug_string();
}
void MnnEngine::print_debug_string()
{
	std::cout << "LITEMNN_DEBUG LogId: " << log_id_ << "\n";
	std::cout << "=============== Input-Dims ==============\n";
	if (input_tensor_)
		input_tensor_->printShape();
	if (dimension_type_ == MNN::Tensor::CAFFE)
		std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
	else if (dimension_type_ == MNN::Tensor::TENSORFLOW)
		std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
	else if (dimension_type_ == MNN::Tensor::CAFFE_C4)
		std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
	std::cout << "=============== Output-Dims ==============\n";
	auto tmp_output_map = mnn_interpreter_->getSessionOutputAll(mnn_session_);
	std::cout << "getSessionOutputAll done!\n";
	for (auto it = tmp_output_map.cbegin(); it != tmp_output_map.cend(); ++it)
	{
		std::cout << "Output: " << it->first << ": ";
		it->second->printShape();
	}
	std::cout << "========================================\n";
}
void MnnEngine::prepare_fake_input()
{
	const float fill_value = 0.5f;

	for (int i=0; i<input_tensor_->elementSize(); ++i)
	{
		input_tensor_->host<float>()[i] = fill_value;
	}
}