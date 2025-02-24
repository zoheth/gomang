#include "ncnn_engine.h"

#include <assert.h>

namespace gomang
{
NcnnEngine::NcnnEngine(const std::string &model_path, const TensorDesc &input_desc, unsigned int num_threads) :
    IEngine(model_path, num_threads, "ncnn"),
    param_path_(model_path + ".param"),
    bin_path_(model_path + ".bin")
{
	initHandler();

	input_info_.push_back(input_desc);
	input_name_         = net_->blobs()[net_->input_indexes()[0]].name;
	input_info_.back().name = input_name_;

	auto extrator = net_->create_extractor();

	ncnn::Mat input = genInputMat(input_desc);
	extrator.input(input_name_.c_str(), input);

	for (auto i : net_->output_indexes())
	{
		ncnn::Mat output;
		auto     &blob = net_->blobs()[i];

		extrator.extract(blob.name.c_str(), output);

		TensorDesc desc;
		desc.name      = blob.name;
		desc.shape     = {1, output.c, output.h, output.w};
		desc.data_type = DataType::kFLOAT32;
		desc.layout    = MemoryLayout::kNCHW;
		desc.mem_type  = MemoryType::kCPU_PINNED;
		desc.alignment = 64;
		output_info_.push_back(desc);

		output_names_.push_back(blob.name);
	}
}

bool NcnnEngine::infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs)
{
	ncnn::Mat input = genInputMat(input_info_[0]);
	memcpy(input.data, inputs[0], input_info_[0].calculateSize());

	assert(output_names_.size() == outputs.size());

	auto extractor = net_->create_extractor();
	extractor.set_light_mode(false);

	extractor.input(input_name_.c_str(), input);

	for (int i=0;i<output_names_.size(); ++i)
	{
		ncnn::Mat output;
		extractor.extract(output_names_[i].c_str(), output);

		auto &desc = output_info_[i];
		memcpy(outputs[i], output.data, desc.calculateSize());
	}


	return true;
}

NcnnEngine::~NcnnEngine() = default;

std::vector<TensorDesc> NcnnEngine::getInputInfo() const
{
	return input_info_;
}

std::vector<TensorDesc> NcnnEngine::getOutputInfo() const
{
	return output_info_;
}

void NcnnEngine::initHandler()
{
	net_                          = std::make_unique<ncnn::Net>();
	net_->opt.num_threads         = static_cast<int>(num_threads_);
	net_->opt.use_vulkan_compute  = false;
	net_->opt.use_fp16_arithmetic = false;

	net_->load_param(param_path_.c_str());
	net_->load_model(bin_path_.c_str());
}

ncnn::Mat NcnnEngine::genInputMat(TensorDesc tensor_desc) const
{
	if (tensor_desc.layout == MemoryLayout::kNHWC)
	{
		return {static_cast<int>(tensor_desc.shape[2]), static_cast<int>(tensor_desc.shape[1]), static_cast<int>(tensor_desc.shape[3])};
	}
	if (tensor_desc.layout == MemoryLayout::kNCHW)
	{
		return {static_cast<int>(tensor_desc.shape[3]), static_cast<int>(tensor_desc.shape[2]), static_cast<int>(tensor_desc.shape[1])};
	}
	return {};
}

}        // namespace gomang