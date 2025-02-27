#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#include "D_dncnn_color_blind_module.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator, iree_hal_device_t **out_device)
{
	// Only register the local-task HAL driver.
	IREE_RETURN_IF_ERROR(iree_hal_local_task_driver_module_register(
	    iree_hal_driver_registry_default()));

	iree_hal_driver_t *driver     = nullptr;
	iree_string_view_t identifier = iree_make_cstring_view("local-task");
	iree_status_t      status     = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), identifier, host_allocator, &driver);

	if (iree_status_is_ok(status))
	{
		status = iree_hal_driver_create_default_device(driver, host_allocator, out_device);
	}

	iree_hal_driver_release(driver);
	return iree_ok_status();
}

iree_const_byte_span_t load_bytecode_module()
{
	const struct iree_file_toc_t *module_file = D_dncnn_color_blind_module_create();
	return iree_make_const_byte_span(module_file->data, module_file->size);
}

iree_hal_buffer_params_t create_buffer_params(
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

int main()
{
	iree_vm_instance_t *instance = nullptr;
	IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
	IREE_CHECK_OK(iree_hal_module_register_all_types(instance));

	iree_hal_device_t *device = nullptr;
	IREE_CHECK_OK(create_sample_device(iree_allocator_system(), &device));
	iree_vm_module_t *hal_module = nullptr;
	IREE_CHECK_OK(iree_hal_module_create(instance, 1, &device, IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_null(), iree_allocator_system(), &hal_module));

	const iree_const_byte_span_t module_data = load_bytecode_module();

	iree_vm_module_t *bytecode_module = nullptr;
	IREE_CHECK_OK(iree_vm_bytecode_module_create(instance, module_data, iree_allocator_null(), iree_allocator_system(), &bytecode_module));

	iree_vm_context_t *context   = nullptr;
	iree_vm_module_t  *modules[] = {hal_module, bytecode_module};
	IREE_CHECK_OK(iree_vm_context_create_with_modules(instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0], iree_allocator_system(), &context));
	iree_vm_module_release(hal_module);
	iree_vm_module_release(bytecode_module);

	const char         kMainFunctionName[] = "module.main_graph";
	iree_vm_function_t main_function;
	IREE_CHECK_OK(iree_vm_context_resolve_function(context, iree_make_cstring_view(kMainFunctionName), &main_function));

	std::vector<float> input_data(1 * 3 * 512 * 512, 0.0f);

	iree_hal_dim_t          shape[4]          = {1, 3, 512, 512};
	iree_hal_buffer_view_t *input_buffer_view = nullptr;
	IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
	    device, iree_hal_device_allocator(device), 4, shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
	    (iree_hal_buffer_params_t) {
	        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
	        .type  = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
	    },
	    iree_make_const_byte_span(input_data.data(), 1 * 3 * 512 * 512 * sizeof(float)), &input_buffer_view));

	iree_vm_list_t *inputs = nullptr;
	IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &inputs));

	iree_vm_ref_t input_buffer_view_ref = iree_hal_buffer_view_move_ref(input_buffer_view);
	IREE_CHECK_OK(iree_vm_list_push_ref_retain(inputs, &input_buffer_view_ref));

	iree_vm_list_t *outputs = nullptr;
	IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/iree_vm_make_undefined_type_def(), 1, iree_allocator_system(), &outputs));

	IREE_CHECK_OK(iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE, nullptr, inputs, outputs, iree_allocator_system()));

	iree_hal_buffer_view_t *ret_buffer_view = iree_vm_list_get_buffer_view_assign(outputs, 0);
	if (ret_buffer_view == nullptr)
	{
		printf("Failed to get output buffer view\n");
		return -1;
	}

	std::vector<float> result(1 * 3 * 512 * 512);
	IREE_CHECK_OK(iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, result.data(), 1 * 3 * 512 * 512 * sizeof(float), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
	                                           iree_infinite_timeout()));

	auto [min_it, max_it] = std::minmax_element(result.begin(), result.end());


	std::cout << "Vector size: " << result.size() << std::endl;
	std::cout << "Min value: " << *min_it << std::endl;
	std::cout << "Max value: " << *max_it << std::endl;

	std::cout << "First elements: " << std::endl;
	int count = std::min(10, static_cast<int>(result.size()));
	for (int i = 0; i < count; i++) {
		std::cout << std::setprecision(6) << result[i] << " ";
	}
	std::cout << std::endl;

	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);
}