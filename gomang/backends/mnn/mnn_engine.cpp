#include "mnn_engine.h"

#include <chrono>
#include <iostream>

namespace gomang
{
MnnEngine::MnnEngine(const std::string &model_path, unsigned int num_threads) :
    IEngine(model_path, num_threads, "MNN")
{
	initHandler();
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

	for (int i = 0; i < input_tensor_->elementSize(); ++i)
	{
		input_tensor_->host<float>()[i] = static_cast<const float *>(inputs[0])[i];
	}
	mnn_interpreter_->runSession(mnn_session_);

	return true;
}
std::vector<TensorDesc> MnnEngine::getInputInfo() const
{
	TensorDesc desc;
	desc.shape     = {input_batch_, input_channel_, input_height_, input_width_};
	desc.data_type = DataType::kFLOAT16;
	desc.layout    = MemoryLayout::kNHWC;
	desc.mem_type  = MemoryType::kCPU_PINNED;
	desc.name      = "input";

	return {desc};
}
std::vector<TensorDesc> MnnEngine::getOutputInfo() const
{
	std::vector<TensorDesc> res;
	auto                    output_map = mnn_interpreter_->getSessionOutputAll(mnn_session_);

	for (auto &it : output_map)
	{
		auto       tensor = it.second;
		TensorDesc desc;
		desc.shape     = {tensor->batch(), tensor->channel(), tensor->height(), tensor->width()};
		desc.data_type = DataType::kFLOAT16;
		desc.layout    = MemoryLayout::kNHWC;
		desc.mem_type  = MemoryType::kCPU_PINNED;
		desc.name      = it.first;

		res.push_back(desc);
	}
	return res;
}

void MnnEngine::initHandler()
{
	mnn_interpreter_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path_.c_str()));

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
}

}        // namespace gomang