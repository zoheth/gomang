#pragma once

#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>

class MnnEngine
{
public:
	explicit MnnEngine(const std::string &_mnn_path, unsigned int _num_threads = 1);

	virtual ~MnnEngine();

	MnnEngine(const MnnEngine &)            = delete;
	MnnEngine(MnnEngine &&)                 = delete;
	MnnEngine &operator=(const MnnEngine &) = delete;
	MnnEngine &operator=(MnnEngine &&)      = delete;

	bool benchmark(int num_warmup = 10, int num_infer = 100);

protected:
	std::shared_ptr<MNN::Interpreter> mnn_interpreter_;
	MNN::Session *mnn_session_{nullptr};
	MNN::Tensor *input_tensor_{nullptr}; // assume single input.
	MNN::ScheduleConfig schedule_config_;
	// std::shared_ptr<MNN::CV::ImageProcess> pretreat_; // init at subclass
	const char *log_id_{nullptr};
	const char *mnn_path_{nullptr};

	const unsigned int num_threads_{}; // initialize at runtime.
	int input_batch_{};
	int input_channel_{};
	int input_height_{};
	int input_width_{};
	int dimension_type_{};
	int num_outputs_{1};

  private:
	void init_handler();
	void print_debug_string();

	void prepare_fake_input();
};
