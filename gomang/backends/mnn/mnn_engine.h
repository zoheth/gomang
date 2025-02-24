#pragma once

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

#include "core/engine.h"

namespace gomang
{
class MnnEngine : public IEngine
{
  public:
	explicit MnnEngine(const std::string &_model_path, unsigned int _num_threads = 1);

	~MnnEngine() override;

	bool infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs) override;

	[[nodiscard]] std::vector<TensorDesc> getInputInfo() const override;
	[[nodiscard]] std::vector<TensorDesc> getOutputInfo() const override;

  protected:
	std::shared_ptr<MNN::Interpreter> mnn_interpreter_;
	MNN::Session                     *mnn_session_{nullptr};
	MNN::Tensor                      *input_tensor_{nullptr};        // assume single input.
	MNN::ScheduleConfig               schedule_config_;
	// std::shared_ptr<MNN::CV::ImageProcess> pretreat_; // init at subclass

	const unsigned int num_threads_{};        // initialize at runtime.
	int                input_batch_{};
	int                input_channel_{};
	int                input_height_{};
	int                input_width_{};
	int                dimension_type_{};
	int                num_outputs_{1};

  private:
	void initHandler();
};
}        // namespace gomang
