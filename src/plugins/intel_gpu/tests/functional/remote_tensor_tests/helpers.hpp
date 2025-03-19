// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#ifndef NOMINMAX
# define NOMINMAX
#endif

#ifdef _WIN32
# include "openvino/runtime/intel_gpu/ocl/dx.hpp"
#elif defined ENABLE_LIBVA
# include "openvino/runtime/intel_gpu/ocl/va.hpp"
#endif
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

namespace {
template <typename T>
T load_entrypoint(const cl_platform_id platform, const std::string name) {
#if defined(__GNUC__) && __GNUC__ < 5
// OCL spec says:
// "The function clGetExtensionFunctionAddressForPlatform returns the address of the extension function named by funcname for a given platform.
//  The pointer returned should be cast to a function pointer type matching the extension function's definition defined in the appropriate extension
//  specification and header file."
// So the pointer-to-object to pointer-to-function cast below is supposed to be valid, thus we suppress warning from old GCC versions.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    T p = reinterpret_cast<T>(clGetExtensionFunctionAddressForPlatform(platform, name.c_str()));
#if defined(__GNUC__) && __GNUC__ < 5
#pragma GCC diagnostic pop
#endif
    if (!p) {
        throw std::runtime_error("clGetExtensionFunctionAddressForPlatform(" + name + ") returned NULL.");
    }
    return p;
}

template <typename T>
T try_load_entrypoint(const cl_platform_id platform, const std::string name) {
    try {
        return load_entrypoint<T>(platform, name);
    } catch (...) {
        return nullptr;
    }
}
}  // namespace

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;
    cl_platform_id _platform;

    clHostMemAllocINTEL_fn _host_mem_alloc_fn = nullptr;
    clMemFreeINTEL_fn _mem_free_fn = nullptr;
    clDeviceMemAllocINTEL_fn _device_mem_alloc_fn = nullptr;
    clEnqueueMemcpyINTEL_fn _enqueue_memcpy_fn = nullptr;
    clGetMemAllocInfoINTEL_fn _get_mem_alloc_info_fn = nullptr;

    void init_extension_functions(cl_platform_id platform) {
        _host_mem_alloc_fn = try_load_entrypoint<clHostMemAllocINTEL_fn>(platform, "clHostMemAllocINTEL");
        _device_mem_alloc_fn = try_load_entrypoint<clDeviceMemAllocINTEL_fn>(platform, "clDeviceMemAllocINTEL");
        _mem_free_fn = try_load_entrypoint<clMemFreeINTEL_fn>(platform, "clMemFreeINTEL");
        _enqueue_memcpy_fn = try_load_entrypoint<clEnqueueMemcpyINTEL_fn>(platform, "clEnqueueMemcpyINTEL");
        _get_mem_alloc_info_fn = try_load_entrypoint<clGetMemAllocInfoINTEL_fn>(platform, "clGetMemAllocInfoINTEL");
    }

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel iGPU OCL device, create context and queue
        {
            const unsigned int refVendorID = 0x8086;
            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[GPU] Can't create OpenCL platform for tests");

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);
            OPENVINO_ASSERT(err == CL_SUCCESS, "[GPU] Can't create OpenCL platform for tests");

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        _platform = id;
                        break;
                    }
                }
            }
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);

            init_extension_functions(_platform);
        }
    }

    explicit OpenCL(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context, true);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);

        cl_int error = clGetDeviceInfo(_device.get(), CL_DEVICE_PLATFORM, sizeof(_platform), &_platform, nullptr);
        if (error) {
            throw std::runtime_error("OpenCL helper failed to retrieve CL_DEVICE_PLATFORM: " + std::to_string(error));
        }

        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);

        init_extension_functions(_platform);
    }

    bool supports_usm() const {
        return _host_mem_alloc_fn != nullptr &&
               _device_mem_alloc_fn != nullptr &&
               _mem_free_fn != nullptr &&
               _enqueue_memcpy_fn != nullptr &&
               _get_mem_alloc_info_fn != nullptr;
    }

    void* allocate_usm_host_buffer(size_t size) const {
        cl_int err_code_ret;
        if (!_device_mem_alloc_fn)
            throw std::runtime_error("[GPU] clHostMemAllocINTEL is nullptr");
        auto ret_ptr = _host_mem_alloc_fn(_context.get(), nullptr, size, 0, &err_code_ret);
        if (err_code_ret != CL_SUCCESS)
            throw std::runtime_error("OpenCL helper failed to allocate USM host memory");
        return ret_ptr;
    }

    void* allocate_usm_device_buffer(size_t size) const {
        cl_int err_code_ret;
        if (!_device_mem_alloc_fn)
            throw std::runtime_error("[GPU] clDeviceMemAllocINTEL is nullptr");
        auto ret_ptr = _device_mem_alloc_fn(_context.get(), _device.get(), nullptr, size, 0, &err_code_ret);
        if (err_code_ret != CL_SUCCESS)
            throw std::runtime_error("OpenCL helper failed to allocate USM device memory");
        return ret_ptr;
    }

    void free_mem(void* usm_ptr) {
        if (!_mem_free_fn)
            throw std::runtime_error("[GPU] clMemFreeINTEL is nullptr");

        _mem_free_fn(_context.get(), usm_ptr);
    }

    cl_int memcpy(const cl::CommandQueue& cpp_queue, void *dst_ptr, const void *src_ptr,
                  size_t bytes_count, bool blocking = true, const std::vector<cl::Event>* wait_list = nullptr, cl::Event* ret_event = nullptr) const {
        if (!_enqueue_memcpy_fn)
            throw std::runtime_error("[GPU] clEnqueueMemcpyINTEL is nullptr");
        cl_event tmp;
        cl_int err = _enqueue_memcpy_fn(
            cpp_queue.get(),
            static_cast<cl_bool>(blocking),
            dst_ptr,
            src_ptr,
            bytes_count,
            wait_list == nullptr ? 0 : static_cast<cl_uint>(wait_list->size()),
            wait_list == nullptr ? nullptr : reinterpret_cast<const cl_event*>(&wait_list->front()),
            ret_event == nullptr ? nullptr : &tmp);

        if (ret_event != nullptr && err == CL_SUCCESS)
            *ret_event = tmp;

        return err;
    }

    cl_unified_shared_memory_type_intel get_allocation_type(const void* usm_ptr) const {
        if (!_get_mem_alloc_info_fn) {
            throw std::runtime_error("[GPU] clGetMemAllocInfoINTEL is nullptr");
        }

        cl_unified_shared_memory_type_intel ret_val;
        size_t ret_val_size;
        _get_mem_alloc_info_fn(_context.get(), usm_ptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(cl_unified_shared_memory_type_intel), &ret_val, &ret_val_size);
        return ret_val;
    }
};
