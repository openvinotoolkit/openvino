// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_resource_interop.hpp"

namespace cldnn::ze {
ze_driver_resource ze_import_driver(cl_device_id ocl_device) {
    ze_ocl_interop& interop = ze_ocl_interop::get_instance();
    auto ze_handle = interop.find_ze_driver(ocl_device);
    const bool is_borrowed = true;
    ze_driver_resource resource(ze_handle, is_borrowed);

    cl_platform_id platform;
    cl_int error = clGetDeviceInfo(ocl_device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr);
    OPENVINO_ASSERT(error == CL_SUCCESS,
                    "[GPU] Failed to get OpenCL platform from device handle (Error code: ",
                    std::to_string(error),
                    ")");
    resource.attach_ocl_handle<ocl_resource_type::platform>(platform, is_borrowed);
    return resource;
}

ze_device_resource ze_import_device(cl_device_id ocl_device) {
    ze_ocl_interop& interop = ze_ocl_interop::get_instance();
    auto ze_handle = interop.get_ze_device(ocl_device);
    const bool is_borrowed = true;
    ze_device_resource resource(ze_handle, is_borrowed);
    resource.attach_ocl_handle<ocl_resource_type::device>(ocl_device, is_borrowed);
    return resource;
}

ze_context_resource ze_import_context(cl_context ocl_context) {
    ze_ocl_interop& interop = ze_ocl_interop::get_instance();
    auto ze_handle = interop.get_ze_context(ocl_context);
    const bool is_borrowed = true;
    ze_context_resource resource(ze_handle, is_borrowed);
    resource.attach_ocl_handle<ocl_resource_type::context>(ocl_context, is_borrowed);
    return resource;
}

ze_command_list_resource ze_import_command_list(cl_command_queue ocl_command_queue) {
    ze_ocl_interop& interop = ze_ocl_interop::get_instance();
    auto ze_handle = interop.get_ze_cmd_list(ocl_command_queue);
    const bool is_borrowed = true;
    ze_command_list_resource resource(ze_handle, is_borrowed);
    resource.attach_ocl_handle<ocl_resource_type::command_queue>(ocl_command_queue, is_borrowed);
    return resource;
}

ze_usm_resource ze_import_usm(cl_mem ocl_buffer, ze_context_resource context) {
    ze_ocl_interop& interop = ze_ocl_interop::get_instance();
    auto ze_usm_ptr = interop.get_ze_usm(ocl_buffer);
    const bool is_borrowed = true;
    ze_usm_resource resource({context.handle(), ze_usm_ptr}, is_borrowed);
    resource.attach_ocl_handle<ocl_resource_type::mem_object>(ocl_buffer, is_borrowed);
    return resource;
}

ze_image_resource ze_import_image(cl_mem ocl_image) {
    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = interop.get_ze_image(ocl_image);
    ze_image_resource resource(ze_handle, true);
    resource.attach_ocl_handle<ocl_resource_type::mem_object>(ocl_image, true);
    return resource;
}

void ze_export_ocl_device(ze_device_resource device) {
    if (device.has_ocl_handle<ocl_resource_type::device>()) {
        return;
    }
    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = device.handle();
    auto ocl_handle = interop.find_ocl_device(ze_handle);
    ocl_owner<ocl_resource_type::device> ocl_owner(ocl_handle);
    device.attach_ocl_handle<ocl_resource_type::device>(std::move(ocl_owner));
}

void ze_export_ocl_context(ze_context_resource context, ze_device_resource device) {
    if (context.has_ocl_handle<ocl_resource_type::context>()) {
        return;
    }

    ze_export_ocl_device(device);
    ocl_context_args context_args;
    context_args.device = device.ocl_handle<ocl_resource_type::device>();

    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = context.handle();
    auto ocl_handle = interop.create_cl_context(ze_handle, context_args);
    ocl_owner<ocl_resource_type::context> ocl_owner(ocl_handle);
    context.attach_ocl_handle<ocl_resource_type::context>(std::move(ocl_owner));
}

void ze_export_ocl_command_queue(ze_command_list_resource cmd_list,
                                 ze_context_resource context,
                                 ze_device_resource device) {
    if (cmd_list.has_ocl_handle<ocl_resource_type::command_queue>()) {
        return;
    }

    ze_export_ocl_context(context, device);
    ocl_queue_args queue_args;
    queue_args.device = device.ocl_handle<ocl_resource_type::device>();
    queue_args.context = context.ocl_handle<ocl_resource_type::context>();

    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = cmd_list.handle();
    auto ocl_handle = interop.create_cl_queue(ze_handle, queue_args);
    ocl_owner<ocl_resource_type::command_queue> ocl_owner(ocl_handle);
    cmd_list.attach_ocl_handle<ocl_resource_type::command_queue>(std::move(ocl_owner));
}

void ze_export_ocl_mem(ze_usm_resource usm,
                       ze_context_resource context,
                       ze_device_resource device,
                       cl_mem_flags flags,
                       size_t size) {
    if (usm.has_ocl_handle<ocl_resource_type::mem_object>()) {
        return;
    }

    ze_export_ocl_context(context, device);
    ocl_buffer_args buffer_args;
    buffer_args.context = context.ocl_handle<ocl_resource_type::context>();
    buffer_args.flags = flags;
    buffer_args.size = size;

    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = usm.handle();
    auto ocl_handle = interop.create_cl_buffer(ze_handle.ptr, buffer_args);
    ocl_owner<ocl_resource_type::mem_object> ocl_owner(ocl_handle);
    usm.attach_ocl_handle<ocl_resource_type::mem_object>(std::move(ocl_owner));
}

void ze_export_ocl_image(ze_image_resource image,
                         ze_context_resource context,
                         ze_device_resource device,
                         cl_mem_flags flags,
                         cl_image_format format,
                         cl_image_desc desc) {
    if (image.has_ocl_handle<ocl_resource_type::mem_object>()) {
        return;
    }

    ze_export_ocl_context(context, device);
    ocl_image_args image_args;
    image_args.context = context.ocl_handle<ocl_resource_type::context>();
    image_args.flags = flags;
    image_args.format = format;
    image_args.desc = desc;

    auto& interop = ze_ocl_interop::get_instance();
    auto ze_handle = image.handle();
    auto ocl_handle = interop.create_cl_image(ze_handle, image_args);
    ocl_owner<ocl_resource_type::mem_object> ocl_owner(ocl_handle);
    image.attach_ocl_handle<ocl_resource_type::mem_object>(std::move(ocl_owner));
}
}  // namespace cldnn::ze
