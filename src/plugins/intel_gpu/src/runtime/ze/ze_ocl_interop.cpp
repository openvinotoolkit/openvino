// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_ocl_interop.hpp"

#include <vector>

namespace cldnn {
namespace ze {
namespace {

static constexpr cl_uint CL_L0_CONTEXT_HANDLE = 0x10021;
static constexpr cl_uint CL_L0_IMMEDIATE_CMD_LIST_HANDLE = 0x10022;
static constexpr cl_uint CL_L0_MEM_OBJ_HANDLE = 0x10024;
static constexpr cl_uint CL_L0_DEVICE_HANDLE = 0x10025;

inline void expect_success(cl_int error, void *handle, const std::string& message) {
    if (error != CL_SUCCESS || handle == nullptr) {
        OPENVINO_THROW(message,
            " (Error code: ", std::to_string(error),
            ", returned handle=", std::to_string((uintptr_t)handle), ")");
    }
}

} // namespace

bool ze_ocl_interop::check_support(ze_device_handle_t ze_device) const {
    try {
        auto ocl_dev = find_ocl_device(ze_device);
        return (ocl_dev != nullptr);
    }
    catch(...) {
        return false;
    }
};

void ze_ocl_interop::init() {
    // Initialize OpenCL and find devices
    auto get_ocl_devices = []() -> std::vector<cl_device_id> {
        cl_int error;
        cl_uint numPlatforms = 0;

        error = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (error != CL_SUCCESS || numPlatforms == 0) {
            return {};
        }
        std::vector<cl_platform_id> platform_ids(numPlatforms);
        error = clGetPlatformIDs(platform_ids.size(), platform_ids.data(), nullptr);
        if (error != CL_SUCCESS) {
            return {};
        }
        std::vector<cl_device_id> device_ids;
        for (auto &platform : platform_ids) {
            cl_uint num_devices = 0;
            error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
            if (error != CL_SUCCESS || num_devices == 0) {
                continue;
            }
            size_t old_size = device_ids.size();
            size_t new_size = old_size + num_devices;
            device_ids.resize(new_size);
            error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device_ids.data() + old_size, nullptr);
            if (error != CL_SUCCESS) {
                continue;
            }
        }
        return device_ids;
    };

    // Initialize Level Zero and find devices
    auto get_ze_devices = []() -> std::vector<ze_device_handle_t> {
        ze_init_driver_type_desc_t desc = { ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, nullptr, ZE_INIT_DRIVER_TYPE_FLAG_GPU };
        ze_result_t error;
        uint32_t driver_count = 0;
        error = zeInitDrivers(&driver_count, nullptr, &desc);
        if (error != ZE_RESULT_SUCCESS || driver_count == 0) {
            return {};
        }
        std::vector<ze_driver_handle_t> drivers(driver_count);
        error = zeInitDrivers(&driver_count, drivers.data(), &desc);
        if (error != ZE_RESULT_SUCCESS) {
            return {};
        }
        std::vector<ze_device_handle_t> devices;
        for (auto &driver : drivers) {
            uint32_t device_count = 0;
            error = zeDeviceGet(driver, &device_count, nullptr);
            if (error != ZE_RESULT_SUCCESS || device_count == 0) {
                continue;
            }
            size_t old_size = devices.size();
            size_t new_size = old_size + device_count;
            devices.resize(new_size);
            error = zeDeviceGet(driver, &device_count, devices.data() + old_size);
            if (error != ZE_RESULT_SUCCESS) {
                continue;
            }
        }
        return devices;
    };
    auto ze_devices = get_ze_devices();
    auto ocl_devices = get_ocl_devices();
    for (auto &ze_device : ze_devices) {
        _device_map[ze_device] = nullptr;
    }
    for (auto &ocl_device : ocl_devices) {
        ze_device_handle_t ze_dev = nullptr;
        try {
            ze_dev = get_ze_device(ocl_device);
        } catch (const std::exception &e) {
            continue;
        }
        _device_map[ze_dev] = ocl_device;
    }
}

cl_device_id ze_ocl_interop::find_ocl_device(ze_device_handle_t ze_device) const {
    auto it = _device_map.find(ze_device);
    if (it == _device_map.end() || it->second == nullptr) {
        OPENVINO_THROW("[GPU] Failed to find matching OCL device for given ZE device");
    }
    return it->second;
}

ze_context_handle_t ze_ocl_interop::get_ze_context(cl_context context) const {
    cl_int error;
    ze_context_handle_t ze_context = nullptr;
    error = clGetContextInfo(context, CL_L0_CONTEXT_HANDLE, sizeof(ze_context_handle_t), &ze_context, nullptr);
    expect_success(error, ze_context, "[GPU] Attempt to extract ZE context from OCL context failed");
    return ze_context;
}

ze_command_list_handle_t ze_ocl_interop::get_ze_cmd_list(cl_command_queue queue) const {
    cl_int error;
    ze_command_list_handle_t ze_cmd_list;
    error = clGetCommandQueueInfo(queue, CL_L0_IMMEDIATE_CMD_LIST_HANDLE, sizeof(ze_command_list_handle_t), &ze_cmd_list, nullptr);
    expect_success(error, ze_cmd_list, "[GPU] Attempt to extract ZE command list from OCL command queue failed");
    return ze_cmd_list;
}

void* ze_ocl_interop::get_ze_usm(cl_mem ocl_mem) const {
    cl_mem mem = reinterpret_cast<cl_mem>(ocl_mem);
    cl_int error;
    void *ze_mem;
    error = clGetMemObjectInfo(mem, CL_L0_MEM_OBJ_HANDLE, sizeof(void*), &ze_mem, nullptr);
    expect_success(error, ze_mem, "[GPU] Attempt to extract ZE usm pointer from OCL buffer failed");
    return ze_mem;
}

ze_image_handle_t ze_ocl_interop::get_ze_image(cl_mem ocl_mem) const {
    cl_mem mem = reinterpret_cast<cl_mem>(ocl_mem);
    cl_int error;
    ze_image_handle_t ze_image;
    error = clGetMemObjectInfo(mem, CL_L0_MEM_OBJ_HANDLE, sizeof(ze_image_handle_t), &ze_image, nullptr);
    expect_success(error, ze_image, "[GPU] Attempt to extract ZE image from OCL buffer failed");
    return ze_image;
}

ze_device_handle_t ze_ocl_interop::get_ze_device(cl_device_id device) const {
    ze_device_handle_t ze_device;
    cl_int error = clGetDeviceInfo(device, CL_L0_DEVICE_HANDLE, sizeof(ze_device_handle_t), &ze_device, nullptr);
    expect_success(error, ze_device, "[GPU] Attempt to extract ZE device from OCL device failed");
    return ze_device;
}

cl_context ze_ocl_interop::create_cl_context(ze_context_handle_t context, const ocl_context_args& args) const {
    cl_context_properties properties[] = {CL_L0_CONTEXT_HANDLE, reinterpret_cast<cl_context_properties>(context), 0};
    constexpr cl_uint num_devices = 1;
    cl_int error;
    auto converted_context = clCreateContext(properties, num_devices, &args.device, nullptr, nullptr, &error);
    expect_success(error, converted_context, "[GPU] Attempt to create CL context from ZE context failed");
    return converted_context;

}

cl_command_queue ze_ocl_interop::create_cl_queue(ze_command_list_handle_t cmd_list, const ocl_queue_args& args) const {
    // Works only for immediate command lists
    cl_queue_properties properties[] = {CL_L0_IMMEDIATE_CMD_LIST_HANDLE, reinterpret_cast<cl_queue_properties>(cmd_list), 0};
    cl_int error;
    auto converted_queue = clCreateCommandQueueWithProperties(args.context, args.device, properties, &error);
    expect_success(error, converted_queue, "[GPU] Attempt to create CL command queue from ZE command list failed");
    return converted_queue;
}

cl_mem ze_ocl_interop::create_cl_buffer(void* usm_ptr, const ocl_buffer_args& args) const {
    cl_mem_properties properties[] = {CL_L0_MEM_OBJ_HANDLE, reinterpret_cast<cl_mem_properties>(usm_ptr), 0};
    cl_int error;
    cl_mem converted_mem = clCreateBufferWithProperties(args.context, properties, args.flags, args.size, nullptr, &error);
    expect_success(error, converted_mem, "[GPU] Attempt to create CL buffer from ZE USM failed");
    return converted_mem;
}

cl_mem ze_ocl_interop::create_cl_image(ze_image_handle_t image, const ocl_image_args &args) const {
    cl_mem_properties properties[] = {CL_L0_MEM_OBJ_HANDLE, reinterpret_cast<cl_mem_properties>(image), 0};
    cl_int error;
    cl_mem converted_mem = clCreateImageWithProperties(args.context, properties, args.flags, &args.format, &args.desc, nullptr, &error);
    expect_success(error, converted_mem, "[GPU] Attempt to create CL image from ZE image failed");
    return converted_mem;
}

}  // namespace ze
}  // namespace cldnn
