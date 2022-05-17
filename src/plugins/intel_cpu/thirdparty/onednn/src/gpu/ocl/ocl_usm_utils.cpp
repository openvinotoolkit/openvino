/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <type_traits>

#include <CL/cl.h>

#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/ocl/ocl_gpu_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace usm {

namespace {

cl_device_id get_ocl_device(engine_t *engine) {
    return utils::downcast<ocl_gpu_engine_t *>(engine)->device();
}

cl_context get_ocl_context(engine_t *engine) {
    return utils::downcast<ocl_gpu_engine_t *>(engine)->context();
}

cl_command_queue get_ocl_queue(stream_t *stream) {
    return utils::downcast<ocl_stream_t *>(stream)->queue();
}

template <typename F>
struct ext_func_t {
    ext_func_t(const char *name) : ext_func_ptrs_(intel_platforms().size()) {
        for (size_t i = 0; i < intel_platforms().size(); ++i) {
            auto p = intel_platforms()[i];
            auto it = ext_func_ptrs_.insert({p, load_ext_func(p, name)});
            assert(it.second);
            MAYBE_UNUSED(it);
        }
    }

    template <typename... Args>
    typename std::result_of<F(Args...)>::type operator()(
            engine_t *engine, Args... args) const {
        cl_platform_id platform;
        cl_int err = clGetDeviceInfo(get_ocl_device(engine), CL_DEVICE_PLATFORM,
                sizeof(platform), &platform, nullptr);
        assert(err == CL_SUCCESS);
        MAYBE_UNUSED(err);
        return ext_func_ptrs_.at(platform)(args...);
    }

private:
    std::unordered_map<cl_platform_id, F> ext_func_ptrs_;

    static F load_ext_func(cl_platform_id platform, const char *name) {
        return reinterpret_cast<F>(
                clGetExtensionFunctionAddressForPlatform(platform, name));
    }

    static const std::vector<cl_platform_id> &intel_platforms() {
        static auto intel_platforms = get_intel_platforms();
        return intel_platforms;
    }

    static std::vector<cl_platform_id> get_intel_platforms() {
        cl_uint num_platforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return {};

        std::vector<cl_platform_id> intel_platforms;
        char vendor_name[128] = {};
        for (cl_platform_id p : platforms) {
            err = clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor_name),
                    vendor_name, nullptr);
            if (err != CL_SUCCESS) continue;
            if (std::string(vendor_name).find("Intel") != std::string::npos)
                intel_platforms.push_back(p);
        }

        // OpenCL can return a list of platforms that contains duplicates.
        std::sort(intel_platforms.begin(), intel_platforms.end());
        intel_platforms.erase(
                std::unique(intel_platforms.begin(), intel_platforms.end()),
                intel_platforms.end());
        return intel_platforms;
    }
};

} // namespace

void *malloc_host(engine_t *engine, size_t size) {
    using clHostMemAllocINTEL_func_t = void *(*)(cl_context, const cl_ulong *,
            size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clHostMemAllocINTEL_func_t> ext_func(
            "clHostMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), nullptr, size, 0, &err);
    assert(utils::one_of(
            err, CL_SUCCESS, CL_OUT_OF_RESOURCES, CL_OUT_OF_HOST_MEMORY));
    return p;
}

void *malloc_device(engine_t *engine, size_t size) {
    using clDeviceMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clDeviceMemAllocINTEL_func_t> ext_func(
            "clDeviceMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(
            err, CL_SUCCESS, CL_OUT_OF_RESOURCES, CL_OUT_OF_HOST_MEMORY));
    return p;
}

void *malloc_shared(engine_t *engine, size_t size) {
    using clSharedMemAllocINTEL_func_t = void *(*)(cl_context, cl_device_id,
            cl_ulong *, size_t, cl_uint, cl_int *);

    if (size == 0) return nullptr;

    static ext_func_t<clSharedMemAllocINTEL_func_t> ext_func(
            "clSharedMemAllocINTEL");
    cl_int err;
    void *p = ext_func(engine, get_ocl_context(engine), get_ocl_device(engine),
            nullptr, size, 0, &err);
    assert(utils::one_of(
            err, CL_SUCCESS, CL_OUT_OF_RESOURCES, CL_OUT_OF_HOST_MEMORY));
    return p;
}

void free(engine_t *engine, void *ptr) {
    using clMemFreeINTEL_func_t = cl_int (*)(cl_context, void *);

    if (!ptr) return;
    static ext_func_t<clMemFreeINTEL_func_t> ext_func("clMemFreeINTEL");
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr);
    assert(err == CL_SUCCESS);
    MAYBE_UNUSED(err);
}

status_t set_kernel_arg_usm(engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value) {
    using clSetKernelArgMemPointerINTEL_func_t
            = cl_int (*)(cl_kernel, cl_uint, const void *);
    static ext_func_t<clSetKernelArgMemPointerINTEL_func_t> ext_func(
            "clSetKernelArgMemPointerINTEL");
    return convert_to_dnnl(ext_func(engine, kernel, arg_index, arg_value));
}

status_t memcpy(stream_t *stream, void *dst, const void *src, size_t size) {
    using clEnqueueMemcpyINTEL_func_t
            = cl_int (*)(cl_command_queue, cl_bool, void *, const void *,
                    size_t, cl_uint, const cl_event *, cl_event *);
    static ext_func_t<clEnqueueMemcpyINTEL_func_t> ext_func(
            "clEnqueueMemcpyINTEL");
    return convert_to_dnnl(ext_func(stream->engine(), get_ocl_queue(stream),
            /* blocking */ CL_FALSE, dst, src, size, 0, nullptr, nullptr));
}

status_t fill(stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size) {
    using clEnqueueMemFillINTEL_func_t
            = cl_int (*)(cl_command_queue, void *, const void *, size_t, size_t,
                    cl_uint, const cl_event *, cl_event *);
    static ext_func_t<clEnqueueMemFillINTEL_func_t> ext_func(
            "clEnqueueMemFillINTEL");
    return convert_to_dnnl(ext_func(stream->engine(), get_ocl_queue(stream),
            ptr, pattern, pattern_size, size, 0, nullptr, nullptr));
}

status_t memset(stream_t *stream, void *ptr, int value, size_t size) {
    return fill(stream, ptr, &value, sizeof(value), size);
}

ocl_usm_kind_t get_pointer_type(engine_t *engine, const void *ptr) {
    using clGetMemAllocInfoINTEL_func_t = cl_int (*)(
            cl_context, const void *, cl_uint, size_t, void *, size_t *);

    // The values are taken from cl_ext.h to avoid dependency on the header.
    static constexpr cl_uint cl_mem_type_unknown_intel = 0x4196;
    static constexpr cl_uint cl_mem_type_host_intel = 0x4197;
    static constexpr cl_uint cl_mem_type_device_intel = 0x4198;
    static constexpr cl_uint cl_mem_type_shared_intel = 0x4199;

    static constexpr cl_uint cl_mem_alloc_type_intel = 0x419A;

    static ext_func_t<clGetMemAllocInfoINTEL_func_t> ext_func(
            "clGetMemAllocInfoINTEL");

    if (!ptr) return ocl_usm_kind_t::unknown;

    cl_uint alloc_type;
    cl_int err = ext_func(engine, get_ocl_context(engine), ptr,
            cl_mem_alloc_type_intel, sizeof(alloc_type), &alloc_type, nullptr);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) return ocl_usm_kind_t::unknown;

    switch (alloc_type) {
        case cl_mem_type_unknown_intel: return ocl_usm_kind_t::unknown;
        case cl_mem_type_host_intel: return ocl_usm_kind_t::host;
        case cl_mem_type_device_intel: return ocl_usm_kind_t::device;
        case cl_mem_type_shared_intel: return ocl_usm_kind_t::shared;
        default: assert(!"unknown alloc type");
    }
    return ocl_usm_kind_t::unknown;
}

} // namespace usm
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
