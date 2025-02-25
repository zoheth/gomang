#pragma once

#include <iree/runtime/api.h>

#include "core/engine.h"

namespace gomang
{
class IreeEngine : public IEngine
{
  public:
	IreeEngine(std::string model_path, unsigned int num_threads = 1);
	~IreeEngine() override;
	bool infer(
	    const std::vector<const void *> &inputs,
	    const std::vector<void *>       &outputs) override;

	[[nodiscard]] std::vector<TensorDesc> getInputInfo() const override;
	[[nodiscard]] std::vector<TensorDesc> getOutputInfo() const override;

  private:
	iree_runtime_instance_t *instance_ = nullptr;
	iree_runtime_session_t  *session_  = nullptr;
	iree_hal_device_t       *device_   = nullptr;

	mutable std::vector<TensorDesc> input_descs_;
	mutable std::vector<TensorDesc> output_descs_;

	iree_status_t initializeIree();
	iree_status_t loadModule();
	void          cleanupIree();

	static iree_hal_element_type_t convertToIreeElementType(DataType type);
	static DataType                convertFromIreeElementType(iree_hal_element_type_t type);

	iree_status_t createBufferView(
	    const void              *data,
	    const TensorDesc        &desc,
	    iree_hal_buffer_view_t **out_buffer_view);

	iree_status_t extractModelInfo();
};
}        // namespace gomang