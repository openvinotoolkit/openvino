// Copyright (C) 2018-2021 Intel Corporation
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

namespace cldnn {
namespace ocl {

ocl_error::ocl_error(cl::Error const& err)
    : std::runtime_error(err.what() + std::string(", error code: ") + std::to_string(err.err())) {}

ocl_engine::ocl_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf)
    : engine(dev, conf) {
    if (runtime_type != runtime_types::ocl) {
        throw std::runtime_error("Invalid runtime type specified for OCL engine. Only OCL runtime is supported");
    }

    auto casted = dynamic_cast<ocl_device*>(dev.get());
    if (!casted)
        throw std::runtime_error("[CLDNN] Invalid device type passed to ocl engine");
    casted->get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    _program_stream.reset(new ocl_stream(*this));
    _usm_helper.reset(new cl::UsmHelper(get_cl_context(), get_cl_device(), use_unified_shared_memory()));
}

const cl::Context& ocl_engine::get_cl_context() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    if (!cl_device)
        throw std::runtime_error("Invalid device type for ocl_engine");
    return cl_device->get_context();
}

const cl::Device& ocl_engine::get_cl_device() const {
    auto cl_device = std::dynamic_pointer_cast<ocl_device>(_device);
    if (!cl_device)
        throw std::runtime_error("Invalid device type for ocl_engine");
    return cl_device->get_device();
}

const cl::UsmHelper& ocl_engine::get_usm_helper() const {
    return *_usm_helper;
}

memory::ptr ocl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    if (layout.bytes_count() > get_device_info().max_alloc_mem_size) {
        throw std::runtime_error("exceeded max size of memory object allocation");
    }

    _memory_pool->add_memory_used(layout.bytes_count());

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
                throw std::runtime_error("out of GPU resources");
            default:
                throw std::runtime_error("GPU buffer allocation failed");
        }
    }
}

memory::ptr ocl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    if (memory.get_engine() != this)
        throw std::runtime_error("trying to reinterpret buffer allocated by a different engine");

    if (new_layout.format.is_image() && !memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret non-image buffer as image");

    if (!new_layout.format.is_image() && memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret image buffer as non-image buffer");

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
            return std::make_shared<ocl::gpu_buffer>(this, new_layout, buf);
        } else {
            throw std::runtime_error("unknown shared object fromat or type");
        }
    }
    catch (const cl::Error& clErr) {
        switch (clErr.err()) {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        case CL_OUT_OF_RESOURCES:
        case CL_OUT_OF_HOST_MEMORY:
        case CL_INVALID_BUFFER_SIZE:
            throw std::runtime_error("out of GPU resources");
        default:
            throw std::runtime_error("GPU buffer allocation failed");
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

stream& ocl_engine::get_program_stream() const {
    return *_program_stream;
}

std::shared_ptr<cldnn::engine> ocl_engine::create(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return std::make_shared<ocl::ocl_engine>(device, runtime_type, configuration);
}

std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return ocl_engine::create(device, runtime_type, configuration);
}

}  // namespace ocl
}  // namespace cldnn
