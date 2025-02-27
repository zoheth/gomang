#pragma once

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#include "core/engine.h"

namespace gomang
{
class IreeEngine : public IEngine
{
  public:
	IreeEngine(std::string model_path, const TensorDesc& input_desc, unsigned int num_threads = 1);
	~IreeEngine() override;
	bool infer(
	    const std::vector<const void *> &inputs,
	    const std::vector<void *>       &outputs) override;

	[[nodiscard]] std::vector<TensorDesc> getInputInfo() const override;
	[[nodiscard]] std::vector<TensorDesc> getOutputInfo() const override;

  private:
	iree_vm_instance_t* instance_{nullptr};
	iree_hal_device_t* device_{nullptr};
	iree_vm_context_t* context_{nullptr};
	iree_vm_function_t main_function_{};

	mutable std::vector<TensorDesc> input_descs_;
	mutable std::vector<TensorDesc> output_descs_;

	iree_const_byte_span_t module_data_{};

	bool initialize();

	iree_status_t createDevice(iree_allocator_t host_allocator);
	iree_const_byte_span_t loadBytecodeModule();



	bool runInference(const void* input_data,
		      void* output_data,
		      bool detect_output = false);

	iree_hal_element_type_t convertDataTypeToIree(DataType data_type) const;

	bool detectOutputInfo();
};
}        // namespace gomang