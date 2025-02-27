#include "iree_engine.h"

#include <filesystem>

#include "D_dncnn_color_blind.h"
#include "SR_msrresnet_x4_psnr.h"
#include "SR_edsr.h"

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

IreeEngine::IreeEngine(std::string model_path, const TensorDesc &input_desc, unsigned int num_threads) :
    IEngine(std::move(model_path), num_threads, "iree")
{
	input_descs_.push_back(input_desc);
	if (!initialize())
	{
		throw std::runtime_error("Failed to initialize IREE engine");
	}

	if (!detectOutputInfo())
	{
		throw std::runtime_error("Failed to detect output information");
	}
}

IreeEngine::~IreeEngine()
{
	iree_hal_device_release(device_);
	iree_vm_context_release(context_);
	iree_vm_instance_release(instance_);
}
bool IreeEngine::infer(const std::vector<const void *> &inputs, const std::vector<void *> &outputs)
{
	if (inputs.empty() || outputs.empty())
	{
		return false;
	}

	return runInference(inputs[0], outputs[0], false);
}
std::vector<TensorDesc> IreeEngine::getInputInfo() const
{
	return input_descs_;
}
std::vector<TensorDesc> IreeEngine::getOutputInfo() const
{
	return output_descs_;
}
bool IreeEngine::initialize()
{
	IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
	                                      iree_allocator_system(), &instance_));
	IREE_CHECK_OK(iree_hal_module_register_all_types(instance_));

	IREE_CHECK_OK(createDevice(iree_allocator_system()));

	iree_vm_module_t *hal_module = nullptr;
	IREE_CHECK_OK(iree_hal_module_create(
	    instance_, 1, &device_, IREE_HAL_MODULE_FLAG_NONE,
	    iree_hal_module_debug_sink_null(), iree_allocator_system(), &hal_module));

	module_data_ = loadBytecodeModule();

	iree_vm_module_t *bytecode_module = nullptr;
	IREE_CHECK_OK(iree_vm_bytecode_module_create(
	    instance_, module_data_, iree_allocator_null(),
	    iree_allocator_system(), &bytecode_module));

	iree_vm_module_t *modules[] = {hal_module, bytecode_module};
	IREE_CHECK_OK(iree_vm_context_create_with_modules(
	    instance_, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules),
	    &modules[0], iree_allocator_system(), &context_));

	// Release module references now that the context holds them
	iree_vm_module_release(hal_module);
	iree_vm_module_release(bytecode_module);

	const char kMainFunctionName[] = "module.main_graph";
	IREE_CHECK_OK(iree_vm_context_resolve_function(
	    context_, iree_make_cstring_view(kMainFunctionName), &main_function_));

	return true;
}
iree_status_t IreeEngine::createDevice(iree_allocator_t host_allocator)
{
	// Register the local-task HAL driver
	IREE_RETURN_IF_ERROR(iree_hal_local_task_driver_module_register(
	    iree_hal_driver_registry_default()));

	// Create driver
	iree_hal_driver_t *driver     = nullptr;
	iree_string_view_t identifier = iree_make_cstring_view("local-task");
	iree_status_t      status     = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), identifier, host_allocator, &driver);

	// Create default device
	if (iree_status_is_ok(status))
	{
		status = iree_hal_driver_create_default_device(driver, host_allocator, &device_);
	}

	iree_hal_driver_release(driver);
	return status;
}
iree_const_byte_span_t IreeEngine::loadBytecodeModule()
{
	const struct iree_file_toc_t *module_file = D_dncnn_color_blind_create();
	return iree_make_const_byte_span(module_file->data, module_file->size);
}
bool IreeEngine::runInference(const void *input_data, void *output_data, bool detect_output)
{
	const auto &input_desc = input_descs_[0];
	size_t      input_size = input_desc.calculateSize();

	std::vector<iree_hal_dim_t> shape(input_desc.shape.size());

	for (size_t i = 0; i < input_desc.shape.size(); i++)
	{
		shape[i] = static_cast<iree_hal_dim_t>(input_desc.shape[i]);
	}

	iree_hal_element_type_t element_type = convertDataTypeToIree(input_desc.data_type);
	if (element_type == IREE_HAL_ELEMENT_TYPE_NONE)
	{
		return false;
	}

	iree_hal_buffer_view_t *input_buffer_view = nullptr;
	IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
	    device_, iree_hal_device_allocator(device_),
	    input_desc.shape.size(), shape.data(), element_type,
	    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
	    (iree_hal_buffer_params_t) {
	        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
	        .type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
	    },
	    iree_make_const_byte_span(input_data, input_size),
	    &input_buffer_view));

	iree_vm_list_t *inputs = nullptr;
	IREE_CHECK_OK(iree_vm_list_create(
	    iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &inputs));

	iree_vm_ref_t input_buffer_view_ref = iree_hal_buffer_view_move_ref(input_buffer_view);
	IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs, &input_buffer_view_ref));

	iree_vm_list_t *outputs = nullptr;
	IREE_CHECK_OK(iree_vm_list_create(
	    iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &outputs));

	IREE_CHECK_OK(iree_vm_invoke(
	    context_, main_function_, IREE_VM_INVOCATION_FLAG_NONE,
	    nullptr, inputs, outputs, iree_allocator_system()));

	iree_hal_buffer_view_t *ret_buffer_view = iree_vm_list_get_buffer_view_assign(outputs, 0);
	if (ret_buffer_view == nullptr)
	{
		iree_vm_list_release(inputs);
		iree_vm_list_release(outputs);
		return false;
	}

	if (detect_output)
	{
		// Extract output information
		iree_hal_element_type_t output_element_type = iree_hal_buffer_view_element_type(ret_buffer_view);
		iree_host_size_t        rank                = iree_hal_buffer_view_shape_rank(ret_buffer_view);

		// Create output TensorDesc
		TensorDesc output_desc;
		output_desc.shape.resize(rank);

		for (iree_host_size_t i = 0; i < rank; ++i)
		{
			output_desc.shape[i] = iree_hal_buffer_view_shape_dim(ret_buffer_view, i);
		}

		// Set data type based on element type
		switch (output_element_type)
		{
			case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
				output_desc.data_type = DataType::kFLOAT32;
				break;
			case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
				output_desc.data_type = DataType::kFLOAT16;
				break;
			case IREE_HAL_ELEMENT_TYPE_SINT_32:
				output_desc.data_type = DataType::kINT32;
				break;
			case IREE_HAL_ELEMENT_TYPE_SINT_8:
				output_desc.data_type = DataType::kINT8;
				break;
			default:
				// Unknown data type
				output_desc.data_type = DataType::kFLOAT32;
				break;
		}

		// Assume same layout and memory type as input for now
		output_desc.layout   = input_desc.layout;
		output_desc.mem_type = input_desc.mem_type;
		output_desc.name     = "output";

		// Store output description
		output_descs_.push_back(output_desc);
	}
	else if (output_data)
	{
		// Copy data from device to host
		const auto &output_desc = output_descs_[0];
		size_t      output_size = output_desc.calculateSize();

		IREE_CHECK_OK(iree_hal_device_transfer_d2h(
		    device_, iree_hal_buffer_view_buffer(ret_buffer_view), 0,
		    output_data, output_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
		    iree_infinite_timeout()));
	}

	// Clean up
	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);

	return true;
}

iree_hal_element_type_t IreeEngine::convertDataTypeToIree(DataType data_type) const
{
	switch (data_type)
	{
		case DataType::kFLOAT32:
			return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
		case DataType::kFLOAT16:
			return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
		case DataType::kINT32:
			return IREE_HAL_ELEMENT_TYPE_SINT_32;
		case DataType::kINT8:
			return IREE_HAL_ELEMENT_TYPE_SINT_8;
		default:
			return IREE_HAL_ELEMENT_TYPE_NONE;
	}
}

bool IreeEngine::detectOutputInfo()
{
	const auto &input_desc = input_descs_[0];
	size_t      input_size = input_desc.calculateSize();

	std::vector<uint8_t> dummy_input(input_size, 0);

	return runInference(dummy_input.data(), nullptr, true);
}
}        // namespace gomang