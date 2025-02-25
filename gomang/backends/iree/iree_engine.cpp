#include "iree_engine.h"

#include <filesystem>

namespace gomang
{

inline iree_hal_buffer_params_t createBufferParams(
    iree_hal_buffer_usage_t  usage,
    iree_hal_memory_access_t access,
    iree_hal_memory_type_t   type)
{
	iree_hal_buffer_params_t params;
	params.usage          = usage;
	params.access         = access;
	params.type           = type;
	params.queue_affinity = 0;
	params.min_alignment  = 0;
	return params;
}

IreeEngine::IreeEngine(std::string model_path, unsigned int num_threads) :
    IEngine(std::move(model_path), num_threads, "IREE")
{
	iree_status_t status = initializeIree();
	if (!iree_status_is_ok(status)) {
		char buffer[256];
		iree_host_size_t actual_length = 0;
		iree_status_format(status, sizeof(buffer), buffer, &actual_length);
		iree_status_free(status);
		throw std::runtime_error(std::string("Failed to initialize IREE: ") + buffer);
	}

	status = loadModule();
	if (!iree_status_is_ok(status)) {
		cleanupIree();
		char buffer[256];
		iree_host_size_t actual_length = 0;
		iree_status_format(status, sizeof(buffer), buffer, &actual_length);
		iree_status_free(status);
		throw std::runtime_error(std::string("Failed to load model: ") + buffer);
	}

	status = extractModelInfo();
	if (!iree_status_is_ok(status)) {
		cleanupIree();
		char buffer[256];
		iree_host_size_t actual_length = 0;
		iree_status_format(status, sizeof(buffer), buffer, &actual_length);
		iree_status_free(status);
		throw std::runtime_error(std::string("Failed to extract model info: ") + buffer);
	}
}
IreeEngine::~IreeEngine()
{
	cleanupIree();
}
bool IreeEngine::infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs)
{
	if (inputs.size() != input_descs_.size()) {
        std::cerr << "Input count mismatch: expected " << input_descs_.size()
                  << ", got " << inputs.size() << std::endl;
        return false;
    }

    if (outputs.size() != output_descs_.size()) {
        std::cerr << "Output count mismatch: expected " << output_descs_.size()
                  << ", got " << outputs.size() << std::endl;
        return false;
    }

    // Initialize the call to the main function
    // Assuming "module.main" is the function name - adapt as needed for your models
    iree_runtime_call_t call;
    iree_status_t status = iree_runtime_call_initialize_by_name(
        session_, iree_make_cstring_view("module.main"), &call);

    if (!iree_status_is_ok(status)) {
        std::cerr << "Failed to initialize function call" << std::endl;
        iree_status_free(status);
        return false;
    }

    // Add inputs to the call
    for (size_t i = 0; i < inputs.size(); ++i) {
        iree_hal_buffer_view_t* input_buffer_view = nullptr;
        status = createBufferView(inputs[i], input_descs_[i], &input_buffer_view);

        if (!iree_status_is_ok(status)) {
            std::cerr << "Failed to create input buffer view for input " << i << std::endl;
            iree_runtime_call_deinitialize(&call);
            iree_status_free(status);
            return false;
        }

        status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_buffer_view);
        iree_hal_buffer_view_release(input_buffer_view);

        if (!iree_status_is_ok(status)) {
            std::cerr << "Failed to push input buffer view for input " << i << std::endl;
            iree_runtime_call_deinitialize(&call);
            iree_status_free(status);
            return false;
        }
    }

    // Invoke the function
    status = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(status)) {
        std::cerr << "Failed to invoke function" << std::endl;
        iree_runtime_call_deinitialize(&call);
        iree_status_free(status);
        return false;
    }

    // Get outputs
    for (size_t i = 0; i < outputs.size(); ++i) {
        iree_hal_buffer_view_t* output_buffer_view = nullptr;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &output_buffer_view);

        if (!iree_status_is_ok(status)) {
            std::cerr << "Failed to get output buffer view for output " << i << std::endl;
            iree_runtime_call_deinitialize(&call);
            iree_status_free(status);
            return false;
        }

        // Copy data from the buffer view to the output buffer
        const TensorDesc& desc = output_descs_[i];
        size_t element_size = desc.getElementsCount();
        size_t total_size = element_size;
        for (const auto& dim : desc.shape) {
            total_size *= dim;
        }

        iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(output_buffer_view);
        iree_device_size_t offset = 0;
        iree_device_size_t length = total_size;

        status = iree_hal_buffer_read_data(buffer, offset, outputs[i], length);
        iree_hal_buffer_view_release(output_buffer_view);

        if (!iree_status_is_ok(status)) {
            std::cerr << "Failed to read output data for output " << i << std::endl;
            iree_runtime_call_deinitialize(&call);
            iree_status_free(status);
            return false;
        }
    }

    iree_runtime_call_deinitialize(&call);
    return true;
}
std::vector<TensorDesc> IreeEngine::getInputInfo() const
{
	return input_descs_;
}
std::vector<TensorDesc> IreeEngine::getOutputInfo() const
{
	return output_descs_;
}
iree_status_t IreeEngine::initializeIree()
{
	// Create the runtime instance
	iree_runtime_instance_options_t instance_options;
	iree_runtime_instance_options_initialize(&instance_options);
	iree_runtime_instance_options_use_all_available_drivers(&instance_options);

	// // Configure threading if needed
	// if (num_threads_ > 0) {
	// 	iree_runtime_instance_options_default_device_params(&instance_options)
	// 		.task.worker_count = num_threads_;
	// }

	IREE_RETURN_IF_ERROR(iree_runtime_instance_create(
		&instance_options, iree_allocator_system(), &instance_));

	// Create the device
	IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
		instance_, iree_make_cstring_view("local-task"), &device_));

	// Create the session
	iree_runtime_session_options_t session_options;
	iree_runtime_session_options_initialize(&session_options);

	IREE_RETURN_IF_ERROR(iree_runtime_session_create_with_device(
		instance_, &session_options, device_,
		iree_runtime_instance_host_allocator(instance_), &session_));

	return iree_ok_status();
}
iree_status_t IreeEngine::loadModule()
{
	if (!std::filesystem::exists(model_path_)) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"Model file not found: %s", model_path_.c_str());
	}

	return iree_runtime_session_append_bytecode_module_from_file(
		session_, model_path_.c_str());
}
void IreeEngine::cleanupIree()
{
	if (session_) {
		iree_runtime_session_release(session_);
		session_ = nullptr;
	}

	if (device_) {
		iree_hal_device_release(device_);
		device_ = nullptr;
	}

	if (instance_) {
		iree_runtime_instance_release(instance_);
		instance_ = nullptr;
	}
}
iree_hal_element_type_t IreeEngine::convertToIreeElementType(DataType type)
{
	switch (type) {
		case DataType::kFLOAT32:
			return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
		case DataType::kFLOAT16:
			return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
		case DataType::kINT32:
			return IREE_HAL_ELEMENT_TYPE_SINT_32;
		case DataType::kINT8:
			return IREE_HAL_ELEMENT_TYPE_SINT_8;
		// case DataType::kUINT8:
		// 	return IREE_HAL_ELEMENT_TYPE_UINT_8;
		default:
			return IREE_HAL_ELEMENT_TYPE_NONE;
	}
}
DataType IreeEngine::convertFromIreeElementType(iree_hal_element_type_t type)
{
	switch (type) {
		case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
			return DataType::kFLOAT32;
		case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
			return DataType::kFLOAT16;
		case IREE_HAL_ELEMENT_TYPE_SINT_32:
			return DataType::kINT32;
		case IREE_HAL_ELEMENT_TYPE_SINT_8:
			return DataType::kINT8;
		// case IREE_HAL_ELEMENT_TYPE_UINT_8:
		// 	return DataType::UINT8;
		default:
			return DataType::kFLOAT32;
	}
}
iree_status_t IreeEngine::createBufferView(const void *data, const TensorDesc &desc, iree_hal_buffer_view_t **out_buffer_view)
{
	iree_hal_dim_t dims[6]; // Assuming max 6D tensors
	size_t rank = desc.shape.size();
	if (rank > 6) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"Tensor rank exceeds maximum supported dimensions (6)");
	}

	for (size_t i = 0; i < rank; ++i) {
		dims[i] = static_cast<iree_hal_dim_t>(desc.shape[i]);
	}

	iree_hal_element_type_t element_type = convertToIreeElementType(desc.data_type);
	if (element_type == IREE_HAL_ELEMENT_TYPE_NONE) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"Unsupported data type");
	}

	size_t element_size = desc.getElementsCount();
	size_t total_size = element_size;
	for (const auto& dim : desc.shape) {
		total_size *= dim;
	}

	return iree_hal_buffer_view_allocate_buffer_copy(
		device_,
		iree_hal_device_allocator(device_),
		rank, dims,
		element_type,
		IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
		createBufferParams(
			IREE_HAL_BUFFER_USAGE_DEFAULT,
			IREE_HAL_MEMORY_ACCESS_ALL,
			IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL
		),
		iree_make_const_byte_span(data, total_size),
		out_buffer_view
	);
}
iree_status_t IreeEngine::extractModelInfo()
{
	// This is model-dependent and would need to be implemented based on your specific model
	// For now, we'll just set some placeholders

	// TODO: Extract real tensor info from the model or accompanying metadata
	// This would typically involve querying the model for its input/output specifications

	// Clear existing descriptors
	input_descs_.clear();
	output_descs_.clear();

	// For demonstration, we'll assume the model expects and produces tensors with certain shapes
	// You would replace this with actual model introspection

	// Example: query function signature or reflection information from IREE

	return iree_ok_status();
}
}        // namespace gomang