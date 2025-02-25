#include "mnn_engine.h"

#include <chrono>
#include <cstring>
#include <iostream>

namespace gomang
{
MnnEngine::MnnEngine(const std::string &model_path, unsigned int num_threads) :
    IEngine(model_path, num_threads, "MNN")
{
	initHandler();

	TensorDesc input_desc;
	input_desc.shape     = {input_batch_, input_channel_, input_height_, input_width_};
	input_desc.data_type = DataType::kFLOAT32;
	input_desc.layout    = MemoryLayout::kNC4HW4;
	input_desc.mem_type  = MemoryType::kCPU_PINNED;
	input_desc.name      = "input";
	input_info_.push_back(input_desc);

	auto output_map = mnn_interpreter_->getSessionOutputAll(mnn_session_);

	for (auto &it : output_map)
	{
		auto       tensor = it.second;
		TensorDesc desc;
		auto       shape = tensor->shape();
		desc.shape       = std::vector<int64_t>(shape.begin(), shape.end());
		desc.data_type   = DataType::kFLOAT32;
		desc.layout      = MemoryLayout::kNC4HW4;
		desc.mem_type    = MemoryType::kCPU_PINNED;
		desc.name        = it.first;

		output_info_.push_back(desc);
	}
}

MnnEngine::~MnnEngine()
{
	mnn_interpreter_->releaseModel();
	if (mnn_session_)
	{
		mnn_interpreter_->releaseSession(mnn_session_);
	}
}
bool MnnEngine::infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs)
{
	if (!mnn_interpreter_ || !mnn_session_)
	{
		std::cerr << "MNN interpreter or session not initialized!" << std::endl;
		return false;
	}
	if (void *inputPtr = input_tensor_->map(MNN::Tensor::MAP_TENSOR_WRITE, input_tensor_->getDimensionType()))
	{
		memcpy(inputPtr, inputs[0], input_info_[0].calculateSize());
		input_tensor_->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input_tensor_->getDimensionType(), inputPtr);
	}
	else
	{
		std::cerr << "Failed to map input tensor!" << std::endl;
		return false;
	}

	mnn_interpreter_->runSession(mnn_session_);

	for (int i = 0; i < output_info_.size(); ++i)
	{
		auto tensor = mnn_interpreter_->getSessionOutput(mnn_session_, output_info_[i].name.c_str());
		if (tensor->size() == output_info_[i].calculateSize())
		{
			void *outputPtr = tensor->map(MNN::Tensor::MAP_TENSOR_READ, tensor->getDimensionType());
			if (outputPtr)
			{
				memcpy(outputs[i], outputPtr, tensor->size());
				tensor->unmap(MNN::Tensor::MAP_TENSOR_READ, tensor->getDimensionType(), outputPtr);
			}
			else
			{
				std::cerr << "Failed to map output tensor! [" << i << "]" << std::endl;
				return false;
			}
		}
		else
		{
			std::cout << "Output tensor size mismatch! [" << i << "]" << std::endl;
			std::cout << "Expected: " << output_info_[i].calculateSize() << " | Got: " << tensor->size() << std::endl;
			return false;
		}
	}

	return true;
}
std::vector<TensorDesc> MnnEngine::getInputInfo() const
{
	return input_info_;
}
std::vector<TensorDesc> MnnEngine::getOutputInfo() const
{
	return output_info_;
}

void MnnEngine::initHandler()
{
	mnn_interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path_.c_str()));

	schedule_config_.numThread = static_cast<int>(num_threads_);
	MNN::BackendConfig backend_config;
	backend_config.precision       = MNN::BackendConfig::Precision_High;
	backend_config.memory          = MNN::BackendConfig::Memory_High;
	schedule_config_.backendConfig = &backend_config;
	schedule_config_.type          = MNN_FORWARD_CUDA;

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
}

}        // namespace gomang