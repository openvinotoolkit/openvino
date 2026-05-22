// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_resource.hpp"
#include "ze_ocl_interop.hpp"

namespace cldnn {
namespace ze {

/// @brief Create resource from external OpenCL resource.
template<ocl_resource_type source_type, ze_resource_type target_type>
struct ze_ocl_importer {
    static_assert(false, "Importer for given resource types is not implemented");
};

template<>
struct ze_ocl_importer<ocl_resource_type::device, ze_resource_type::driver> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::device;
    static constexpr ze_resource_type target_type = ze_resource_type::driver;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    resource_t operator()(ocl_handle_t handle) const {
        ze_ocl_interop &interop = ze_ocl_interop::get_instance();
        auto ze_handle = interop.find_ze_driver(handle);
        resource_t resource(ze_handle, true);
        cl_platform_id platform;
        cl_int error = clGetDeviceInfo(handle, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr);
        OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] Failed to get OpenCL platform from device handle (Error code: ", std::to_string(error), ")");
        resource.attach_ocl_handle<ocl_resource_type::platform>(platform, true);
        return resource;
    }
};

template<>
struct ze_ocl_importer<ocl_resource_type::device, ze_resource_type::device> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::device;
    static constexpr ze_resource_type target_type = ze_resource_type::device;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    resource_t operator()(ocl_handle_t handle) const {
        ze_ocl_interop &interop = ze_ocl_interop::get_instance();
        auto ze_handle = interop.get_ze_device(handle);
        resource_t resource(ze_handle, true);
        resource.attach_ocl_handle<source_type>(handle, true);
        return resource;
    }
};

template<>
struct ze_ocl_importer<ocl_resource_type::context, ze_resource_type::context> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::context;
    static constexpr ze_resource_type target_type = ze_resource_type::context;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    resource_t operator()(ocl_handle_t handle) const {
        ze_ocl_interop &interop = ze_ocl_interop::get_instance();
        auto ze_handle = interop.get_ze_context(handle);
        resource_t resource(ze_handle, true);
        resource.attach_ocl_handle<source_type>(handle, true);
        return resource;
    }
};

template<>
struct ze_ocl_importer<ocl_resource_type::command_queue, ze_resource_type::command_list> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::command_queue;
    static constexpr ze_resource_type target_type = ze_resource_type::command_list;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    resource_t operator()(ocl_handle_t handle) const {
        ze_ocl_interop &interop = ze_ocl_interop::get_instance();
        auto ze_handle = interop.get_ze_cmd_list(handle);
        resource_t resource(ze_handle, true);
        resource.attach_ocl_handle<source_type>(handle, true);
        return resource;
    }
};

template<>
struct ze_ocl_importer<ocl_resource_type::mem_object, ze_resource_type::usm_memory> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::mem_object;
    static constexpr ze_resource_type target_type = ze_resource_type::usm_memory;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    struct args_t {
        ze_context_handle_t context;
    };

    ze_ocl_importer(args_t args) : _args(args) {}
    resource_t operator()(ocl_handle_t handle) const {
        auto &interop = ze_ocl_interop::get_instance();
        auto ze_usm_ptr = interop.get_ze_usm(handle);
        resource_t resource({_args.context, ze_usm_ptr}, true);
        resource.attach_ocl_handle<source_type>(handle, true);
        return resource;
    }
private:
    args_t _args;
};

template<>
struct ze_ocl_importer<ocl_resource_type::mem_object, ze_resource_type::image> {
public:
    static constexpr ocl_resource_type source_type = ocl_resource_type::mem_object;
    static constexpr ze_resource_type target_type = ze_resource_type::image;
    using ocl_handle_t = typename ocl_resource_info<source_type>::handle_t;
    using resource_t = ze_resource<target_type>;
    resource_t operator()(ocl_handle_t handle) const {
        auto &interop = ze_ocl_interop::get_instance();
        auto ze_handle = interop.get_ze_image(handle);
        resource_t resource(ze_handle, true);
        resource.attach_ocl_handle<source_type>(handle, true);
        return resource;
    }
};

}  // namespace ze
}  // namespace cldnn
