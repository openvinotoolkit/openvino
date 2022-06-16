// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_engine.hpp"
#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "ocl_stream.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
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
#endif

namespace cldnn {
namespace ocl {

ocl_error::ocl_error(cl::Error const& err)
    : std::runtime_error(err.what() + std::string(", error code: ") + std::to_string(err.err())) {}

ocl_engine::ocl_engine(const device::ptr dev, runtime_types runtime_type,
            const engine_configuration& conf, const InferenceEngine::ITaskExecutor::Ptr task_executor)
    : engine(dev, conf, task_executor) {
    if (runtime_type != runtime_types::ocl) {
        IE_THROW() << "Invalid runtime type specified for OCL engine. Only OCL runtime is supported";
    }

    auto casted = dynamic_cast<ocl_device*>(dev.get());
    if (!casted)
        IE_THROW() << "[CLDNN] Invalid device type passed to ocl engine";
    casted->get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    _usm_helper.reset(new cl::UsmHelper(get_cl_context(), get_cl_device(), use_unified_shared_memory()));

#ifdef ENABLE_ONEDNN_FOR_GPU
    _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ocl_interop::make_engine(casted->get_device().get(), casted->get_context().get()));
#endif
    _program_stream.reset(new ocl_stream(*this));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::engine& ocl_engine::get_onednn_engine() const {
    if (!_onednn_engine)
        IE_THROW() << "[GPU] onednn engine is nullptr";
    return *_onednn_engine;
}
#endif

const cl::Context& ocl_engine::get_cl_context() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    if (!cl_device)
        IE_THROW() << "Invalid device type for ocl_engine";
    return cl_device->get_context();
}

const cl::Device& ocl_engine::get_cl_device() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    if (!cl_device)
        IE_THROW() << "Invalid device type for ocl_engine";
    return cl_device->get_device();
}

const cl::UsmHelper& ocl_engine::get_usm_helper() const {
    return *_usm_helper;
}

memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    if (layout.bytes_count() > get_device_info().max_alloc_mem_size) {
        std::stringstream ss;
        ss << "Exceeded max size of memory object allocation: "
            << "Requested " << layout.bytes_count() << " bytes "
            << "but max alloc size is " << get_device_info().max_alloc_mem_size << " bytes";
        IE_THROW() << ss.str();
    }

    auto used_mem = get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    if (layout.bytes_count() + used_mem > get_max_memory_size()) {
        std::stringstream ss;
        ss << "Exceeded max size of memory allocation: "
            << "Required " << layout.bytes_count() + used_mem << " bytes "
            << "but max alloc size is " << get_max_memory_size() << " bytes";
        IE_THROW() << ss.str();
    }

    if (type != allocation_type::cl_mem && !supports_allocation(type)) {
        std::ostringstream type_str;
        type_str << type;
        IE_THROW() << "Unsupported allocation type " + type_str.str();
    }

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
            res->fill(get_program_stream());
        }

        return res;
    } catch (const cl::Error& clErr) {
        switch (clErr.err()) {
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            case CL_OUT_OF_RESOURCES:
            case CL_OUT_OF_HOST_MEMORY:
            case CL_INVALID_BUFFER_SIZE:
                IE_THROW() << "out of GPU resources";
            default:
                IE_THROW() << "GPU buffer allocation failed";
        }
    }
}

memory::ptr ocl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    if (memory.get_engine() != this)
        IE_THROW() << "trying to reinterpret buffer allocated by a different engine";

    if (new_layout.format.is_image() && !memory.get_layout().format.is_image())
        IE_THROW() << "trying to reinterpret non-image buffer as image : " << memory.get_layout().format.to_string()
                   << " --> " << new_layout.format.to_string();

    if (!new_layout.format.is_image() && memory.get_layout().format.is_image())
        IE_THROW() << "trying to reinterpret image buffer as non-image buffer : "
                   << memory.get_layout().format.to_string() << " --> " << new_layout.format.to_string();

    try {
        if (new_layout.format.is_image_2d()) {
           return std::make_shared<ocl::gpu_image2d>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_image2d&>(memory).get_buffer());
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            return std::make_shared<ocl::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ocl::gpu_usm&>(memory).get_buffer(),
                                     memory.get_allocation_type());
        } else {
           return std::make_shared<ocl::gpu_buffer>(this,
                                    new_layout,
                                    reinterpret_cast<const ocl::gpu_buffer&>(memory).get_buffer());
        }
    } catch (cl::Error const& err) {
        throw ocl::ocl_error(err);
    }
}

memory::ptr ocl_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
   try {
        if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
            cl::Image2D img(static_cast<cl_mem>(params.mem), true);
            return std::make_shared<ocl::gpu_image2d>(this, new_layout, img);
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
            if (actual_mem_size < requested_mem_size) {
                IE_THROW() << "[GPU] shared buffer has smaller size (" << std::to_string(actual_mem_size) <<
                                  ") than specified layout (" << std::to_string(requested_mem_size) << ")";
            }
            return std::make_shared<ocl::gpu_buffer>(this, new_layout, buf);
        } else if (params.mem_type == shared_mem_type::shared_mem_usm) {
            cl::UsmMemory usm_buffer(get_usm_helper(), params.mem);
            auto actual_mem_size = get_usm_helper().get_usm_allocation_size(usm_buffer.get());
            auto requested_mem_size = new_layout.bytes_count();
            if (actual_mem_size < requested_mem_size) {
                IE_THROW() << "[GPU] shared USM buffer has smaller size (" << std::to_string(actual_mem_size)
                           << ") than specified layout (" << std::to_string(requested_mem_size) << ")";
            }
            return std::make_shared<ocl::gpu_usm>(this, new_layout, usm_buffer);
        } else {
            IE_THROW() << "unknown shared object fromat or type";
        }
    }
    catch (const cl::Error& clErr) {
        switch (clErr.err()) {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_INVALID_BUFFER_SIZE:
            IE_THROW() << "out of GPU resources";
        default:
            IE_THROW() << "GPU buffer allocation failed";
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

bool ocl_engine::extension_supported(std::string extension) const {
    return _extensions.find(extension) != std::string::npos;
}

stream::ptr ocl_engine::create_stream() const {
    return std::make_shared<ocl_stream>(*this);
}

stream::ptr ocl_engine::create_stream(void* handle) const {
    return std::make_shared<ocl_stream>(*this, handle);
}

stream& ocl_engine::get_program_stream() const {
    return *_program_stream;
}

std::shared_ptr<cldnn::engine> ocl_engine::create(const device::ptr device, runtime_types runtime_type,
                            const engine_configuration& configuration, const InferenceEngine::ITaskExecutor::Ptr task_executor) {
    return std::make_shared<ocl::ocl_engine>(device, runtime_type, configuration, task_executor);
}

std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type,
                            const engine_configuration& configuration, const InferenceEngine::ITaskExecutor::Ptr task_executor) {
    return ocl_engine::create(device, runtime_type, configuration, task_executor);
}

}  // namespace ocl
}  // namespace cldnn
