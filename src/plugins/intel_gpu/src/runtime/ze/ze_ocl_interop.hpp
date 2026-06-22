// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_ocl_common.hpp"

#include <map>
#include <mutex>

namespace cldnn {
namespace ze {
struct ocl_context_args {
    cl_device_id device;
};
struct ocl_queue_args {
    cl_context context;
    cl_device_id device;
};

struct ocl_buffer_args {
    cl_context context;
    cl_mem_flags flags;
    size_t size;
};

struct ocl_image_args {
    cl_context context;
    cl_mem_flags flags;
    cl_image_format format;
    cl_image_desc desc;
};

/// @brief Provides functions to convert between Level Zero and OpenCL handles.
///
/// Converting from OpenCL to Level Zero generally works by extracting internal Level Zero handles from OpenCL objects.
/// Converting from Level Zero to OpenCL generally works by creating new OpenCL objects and passing to them Level Zero handles.
/// Both approaches enable user to use the same entity through Level Zero and OpenCL APIs.
///
/// Functions named "create_*" work by creating new objects.
/// Handles returned by these functions must be released by the caller using the appropriate release functions.
///
/// Functions named "get_*" work by extracting data from already existing objects.
/// Handles returned by these functions are borrowed and should not be released.
///
/// Functions named "find_*" work by matching already existing objects.
/// Handles returned by these functions should not be released.
///
/// Conversion functions require both Level Zero and OpenCL to be initialized beforehand.
/// All functions will either succeed and return a valid resource handle or throw an exception.
class ze_ocl_interop {
public:
    static ze_ocl_interop& get_instance() {
        static ze_ocl_interop instance;
        static std::once_flag init_flag;
        std::call_once(init_flag, []() {
            instance.init();
        });

        return instance;
    }
    ze_ocl_interop(const ze_ocl_interop&) = delete;
    ze_ocl_interop& operator=(const ze_ocl_interop&) = delete;

    bool check_support(ze_device_handle_t ze_device) const;

    cl_device_id find_ocl_device(ze_device_handle_t ze_device) const;
    ze_driver_handle_t find_ze_driver(cl_device_id ocl_device) const;

    ze_context_handle_t get_ze_context(cl_context context) const;
    ze_command_list_handle_t get_ze_cmd_list(cl_command_queue queue) const;
    void* get_ze_usm(cl_mem ocl_mem) const;
    ze_image_handle_t get_ze_image(cl_mem ocl_mem) const;
    ze_device_handle_t get_ze_device(cl_device_id device) const;

    cl_context create_cl_context(ze_context_handle_t context, const ocl_context_args& args) const;
    cl_command_queue create_cl_queue(ze_command_list_handle_t cmd_list, const ocl_queue_args& args) const;
    cl_mem create_cl_buffer(void* usm_ptr, const ocl_buffer_args& args) const;
    cl_mem create_cl_image(ze_image_handle_t image, const ocl_image_args& args) const;
protected:
    ze_ocl_interop() = default;
    /// @brief Initialize OpenCL and L0 drivers and fill device_map
    void init();

    // L0 driver/device handles and OCL platform/device ids are global read only objects so we can cache them
    std::map<ze_device_handle_t, cl_device_id> _device_map;
};

}  // namespace ze
}  // namespace cldnn
