// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"  // For CACHE_PAGE_SIZE
#include "openvino/runtime/intel_gpu/remote_properties.hpp"

#include "ocl_kernel.hpp"
#include "ocl_kernel_builder.hpp"
#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "ocl_stream.hpp"
#include "ocl_engine_factory.hpp"
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

// static class memebers - pointers to dynamically obtained OpenCL extension functions
cl::PFN_clEnqueueAcquireMediaSurfacesINTEL cl::SharedSurfLock::pfn_acquire = NULL;
cl::PFN_clEnqueueReleaseMediaSurfacesINTEL cl::SharedSurfLock::pfn_release = NULL;
cl::PFN_clCreateFromMediaSurfaceINTEL cl::ImageVA::pfn_clCreateFromMediaSurfaceINTEL = NULL;
#ifdef _WIN32
cl::PFN_clCreateFromD3D11Buffer cl::BufferDX::pfn_clCreateFromD3D11Buffer = NULL;
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include "intel_gpu/runtime/file_util.hpp"
#endif

namespace cldnn {
namespace ocl {

OPENVINO_SUPPRESS_DEPRECATED_START
ocl_error::ocl_error(cl::Error const& err)
    : ov::Exception("[GPU] " + std::string(err.what()) + std::string(", error code: ") + std::to_string(err.err())) {}
OPENVINO_SUPPRESS_DEPRECATED_END

namespace {
cl_platform_id get_platform_id_for_device(const cl::Device& device) {
    cl_platform_id platform = nullptr;
    cl_int err = clGetDeviceInfo(device.get(), CL_DEVICE_PLATFORM, sizeof(platform), &platform, nullptr);
    OPENVINO_ASSERT(err == CL_SUCCESS && platform != nullptr,
                    "[GPU] Failed to retrieve CL_DEVICE_PLATFORM, error: ", err);
    return platform;
}
}  // namespace

ocl_engine::ocl_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::ocl, "[GPU] Invalid runtime type specified for OCL engine. Only OCL runtime is supported");

    auto casted = dynamic_cast<ocl_device*>(dev.get());
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type passed to ocl engine");
    casted->get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    _service_stream.reset(new ocl_stream(*this, ExecutionConfig()));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void ocl_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        auto casted = std::dynamic_pointer_cast<ocl_device>(_device);
        OPENVINO_ASSERT(casted, "[GPU] Invalid device type stored in ocl_engine");
#ifdef OV_GPU_WITH_ZE_RT
        OPENVINO_THROW("[GPU] Using OCL OneDNN API with Level Zero runtime");
#else
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get()));
#endif
    }
}
#endif

const cl::Context& ocl_engine::get_cl_context() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return cl_device->get_context();
}

const cl::Device& ocl_engine::get_cl_device() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return cl_device->get_device();
}

const cl::UsmHelper& ocl_engine::get_usm_helper() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return cl_device->get_usm_helper();
}

allocation_type ocl_engine::detect_usm_allocation_type(const void* memory) const {
    return use_unified_shared_memory() ? ocl::gpu_usm::detect_allocation_type(this, memory)
                                       : allocation_type::unknown;
}

memory::ptr ocl_engine::import_buffer(const layout& layout, ov::intel_gpu::os_handle_param external_handle) {
#ifdef __linux__
    OPENVINO_ASSERT(external_handle >= 0, "[GPU] External memory handle must be a valid file descriptor");
#else
    OPENVINO_ASSERT(external_handle != nullptr, "[GPU] External memory handle must not be null");
#endif
    OPENVINO_ASSERT(extension_supported("cl_khr_external_memory"),
                    "[GPU] Selected OpenCL device does not advertise cl_khr_external_memory; "
                    "external memory import is not supported");

#ifndef CL_VERSION_3_0
    OPENVINO_THROW("[GPU] External memory import is not supported on this platform");
#else
#ifdef _WIN32
    constexpr auto handle_type_token = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
#elif defined(__linux__)
    constexpr auto handle_type_token = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
#else
    OPENVINO_THROW("[GPU] External memory import is not supported on this platform");
#endif

    cl_mem_properties props[] = {
        static_cast<cl_mem_properties>(handle_type_token),
#ifdef __linux__
        static_cast<cl_mem_properties>(external_handle),
#else
        static_cast<cl_mem_properties>(reinterpret_cast<intptr_t>(external_handle)),
#endif
        0,
    };

    cl_int errcode = CL_SUCCESS;
    auto cl_ctx = static_cast<cl_context>(get_user_context());
    OPENVINO_ASSERT(cl_ctx != nullptr, "[GPU] OpenCL context is null while importing external buffer");
    const auto byte_size = layout.bytes_count();
    cl_mem imported = clCreateBufferWithProperties(cl_ctx, props, CL_MEM_READ_WRITE, byte_size, nullptr, &errcode);
    OPENVINO_ASSERT(errcode == CL_SUCCESS && imported != nullptr,
                    "[GPU] Failed to import external memory handle via clCreateBufferWithProperties, error: ",
                    errcode);

    cl_platform_id platform = get_platform_id_for_device(get_cl_device());
    auto& svc_stream = downcast<ocl_stream>(get_service_stream());
    cl_command_queue q = svc_stream.get_cl_queue().get();
    cl_int acquire_err = cl::ExternalMemoryHelper::acquire(platform, q, imported);
    if (acquire_err != CL_SUCCESS) {
        clReleaseMemObject(imported);
        OPENVINO_THROW("[GPU] clEnqueueAcquireExternalMemObjectsKHR failed or unavailable, error: ", acquire_err);
    }
    clFinish(q);
    cl::Buffer buf(imported, true);
    auto memory = std::make_shared<ocl::gpu_buffer_from_handle>(this, layout, buf, nullptr);
    clReleaseMemObject(imported);
    return memory;
#endif
}

void ocl_engine::release_external_memory(cl_mem mem) const {
    cl_platform_id platform = get_platform_id_for_device(get_cl_device());
    auto& opencl_stream = downcast<ocl_stream>(get_service_stream());
    cl_command_queue q = opencl_stream.get_cl_queue().get();
    // If the extension entrypoint is missing, the cl_mem refcount drop on dtor will still proceed.
    cl::ExternalMemoryHelper::release(platform, q, mem);
    clFinish(q);
}

memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res = nullptr;
        if (layout.format.is_image_2d()) {
            res = std::make_shared<ocl::gpu_image2d>(this, layout);
        } else if (type == allocation_type::cl_mem) {
            res = std::make_shared<ocl::gpu_buffer>(this, layout);
        } else {
            res = std::make_shared<ocl::gpu_usm>(this, layout, type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const cl::Error& clErr) {
        switch (clErr.err()) {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            case CL_OUT_OF_RESOURCES:
            case CL_OUT_OF_HOST_MEMORY:
            case CL_INVALID_BUFFER_SIZE:
                OPENVINO_THROW("[GPU] out of GPU resources");
            default:
                OPENVINO_THROW("[GPU] buffer allocation failed");
        }
    }
}
memory::ptr ocl_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to create a subbuffer from a buffer allocated by a different engine");
    try {
        if (new_layout.format.is_image_2d()) {
            OPENVINO_NOT_IMPLEMENTED;
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            auto& new_buf = reinterpret_cast<const ocl::gpu_usm&>(memory);
            auto ptr = new_buf.get_buffer().get();
            auto sub_buffer = cl::UsmMemory(get_usm_helper(), ptr, byte_offset);

            return std::make_shared<ocl::gpu_usm>(this,
                                     new_layout,
                                     sub_buffer,
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
        } else {
            auto buffer = reinterpret_cast<const ocl::gpu_buffer&>(memory).get_buffer();
            cl_buffer_region sub_buffer_region = {byte_offset, new_layout.bytes_count()};
            auto sub_buffer = buffer.createSubBuffer({}, CL_BUFFER_CREATE_TYPE_REGION, &sub_buffer_region);

            return std::make_shared<ocl::gpu_buffer>(this,
                                     new_layout,
                                     sub_buffer,
                                     memory.get_mem_tracker());
        }
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

memory_ptr ocl_engine::create_mmap_hostbuffer(const void* mmapped_address, size_t data_size, allocation_type _allocation_type, const layout output_layout) {
    auto tracker = std::make_shared<MemoryTracker>(this,
                                                   const_cast<void*>(mmapped_address),  // Point directly to mmap'd memory
                                                   data_size,
                                                   _allocation_type);
    std::uintptr_t mmap_address = reinterpret_cast<std::uintptr_t>(mmapped_address);
    std::uintptr_t aligned_addr = mmap_address & ~(static_cast<std::uintptr_t>(cldnn::CACHE_PAGE_SIZE) - 1);
    void* mmap_aligned_address = reinterpret_cast<void*>(aligned_addr);

    cl_int err = CL_SUCCESS;
    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
#ifdef CL_MEM_FORCE_HOST_MEMORY_INTEL
    flags |= CL_MEM_FORCE_HOST_MEMORY_INTEL;
#endif
    cl::Buffer buffer(get_cl_context(), flags, data_size, mmap_aligned_address, &err);
    OPENVINO_ASSERT(err == CL_SUCCESS, "clcreatebuffer with CL_MEM_USE_HOST_PTR and CL_MEM_FORCE_HOST_MEMORY_INTEL failed!");
    return std::make_shared<ocl::gpu_buffer>(this, output_layout, buffer, tracker);
}

memory::ptr ocl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    try {
        bool from_memory_pool = memory.from_memory_pool;
        memory::ptr reinterpret_memory = nullptr;
        if (new_layout.format.is_image_2d()) {
           reinterpret_memory = std::make_shared<ocl::gpu_image2d>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_image2d&>(memory).get_buffer(),
                                     memory.get_mem_tracker());
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
           reinterpret_memory = std::make_shared<ocl::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_usm&>(memory).get_buffer(),
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
        } else {
           reinterpret_memory = std::make_shared<ocl::gpu_buffer>(this,
                                    new_layout,
                                    reinterpret_cast<const ocl::gpu_buffer&>(memory).get_buffer(),
                                    memory.get_mem_tracker());
        }
        reinterpret_memory->from_memory_pool = from_memory_pool;
        return reinterpret_memory;
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

memory::ptr ocl_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
   try {
        if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
            cl::Image2D img(static_cast<cl_mem>(params.mem), true);
            return std::make_shared<ocl::gpu_image2d>(this, new_layout, img, nullptr);
        } else if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_vasurface) {
            return std::make_shared<ocl::gpu_media_buffer>(this, new_layout, params);
#ifdef _WIN32
        } else if (params.mem_type == shared_mem_type::shared_mem_dxbuffer) {
            return std::make_shared<ocl::gpu_dx_buffer>(this, new_layout, params);
#endif
        } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
            cl::Buffer buf(static_cast<cl_mem>(params.mem), true);
            auto actual_mem_size = buf.getInfo<CL_MEM_SIZE>();
            auto requested_mem_size = new_layout.bytes_count();
            OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
            return std::make_shared<ocl::gpu_buffer>(this, new_layout, buf, nullptr);
        } else if (params.mem_type == shared_mem_type::shared_mem_usm) {
            cl::UsmMemory usm_buffer(get_usm_helper(), params.mem);
            auto actual_mem_size = get_usm_helper().get_usm_allocation_size(usm_buffer.get());
            auto requested_mem_size = new_layout.bytes_count();
            OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared USM buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
            return std::make_shared<ocl::gpu_usm>(this, new_layout, usm_buffer, nullptr);
        } else {
            OPENVINO_THROW("[GPU] unknown shared object fromat or type");
        }
    }
    catch (const cl::Error& clErr) {
        switch (clErr.err()) {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_INVALID_BUFFER_SIZE:
            OPENVINO_THROW("[GPU] out of GPU resources");
        default:
            OPENVINO_THROW("[GPU] buffer allocation failed");
        }
    }
}

bool ocl_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
        return (reinterpret_cast<const ocl::gpu_buffer&>(mem1).get_buffer() ==
                reinterpret_cast<const ocl::gpu_buffer&>(mem2).get_buffer());
    else
        return (reinterpret_cast<const ocl::gpu_usm&>(mem1).get_buffer() ==
                reinterpret_cast<const ocl::gpu_usm&>(mem2).get_buffer());
}

void* ocl_engine::get_user_context() const {
    auto& cl_device = downcast<ocl_device>(*_device);
    return static_cast<void*>(cl_device.get_context().get());
}

std::shared_ptr<kernel_builder> ocl_engine::create_kernel_builder() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    OPENVINO_ASSERT(cl_device, "[GPU] Invalid device type for ocl_engine");
    return std::make_shared<ocl_kernel_builder>(*cl_device);
}

bool ocl_engine::extension_supported(std::string extension) const {
    return _extensions.find(extension) != std::string::npos;
}

stream::ptr ocl_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ocl_stream>(*this, config);
}

stream::ptr ocl_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<ocl_stream>(*this, config, handle);
}

std::shared_ptr<cldnn::engine> ocl_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ocl::ocl_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type) {
    return ocl_engine::create(device, runtime_type);
}

}  // namespace ocl
}  // namespace cldnn




