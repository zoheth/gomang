#pragma once

#include <ncnn/layer.h>
#include <ncnn/net.h>

#include "core/engine.h"

namespace gomang
{
class NcnnEngine : public IEngine
{
  public:
	explicit NcnnEngine(const std::string &model_path,const TensorDesc& input_desc, unsigned int num_threads = 1);

	~NcnnEngine() override;

	bool infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs) override;

	[[nodiscard]] std::vector<TensorDesc> getInputInfo() const override;
	[[nodiscard]] std::vector<TensorDesc> getOutputInfo() const override;

  protected:
	std::unique_ptr<ncnn::Net> net_{nullptr};

	std::string param_path_{};
	std::string bin_path_{};

	std::string input_name_{}; // input blob name
	std::vector<std::string> output_names_{}; // output blob name

	std::vector<TensorDesc> input_info_;
	std::vector<TensorDesc> output_info_;

  private:
	void initHandler();

	[[nodiscard]] ncnn::Mat genInputMat(TensorDesc tensor_desc) const;
};
}        // namespace gomang