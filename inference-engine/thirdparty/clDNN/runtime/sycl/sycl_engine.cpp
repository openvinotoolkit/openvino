// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_engine.hpp"
#include "sycl_memory.hpp"
#include "sycl_stream.hpp"
#include "sycl_engine_factory.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {
namespace sycl {

sycl_engine::sycl_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf)
    : engine(dev, conf), _runtime_type(runtime_type) {
    auto casted = dynamic_cast<sycl_device*>(dev.get());
    if (!casted)
        throw std::runtime_error("[CLDNN] Invalid device type passed to sycl engine");
    _extensions = casted->get_device().get_info<cl::sycl::info::device::extensions>();

    _program_stream.reset(new sycl_stream(*this));
}

const cl::sycl::context& sycl_engine::get_sycl_context() const {
    auto cl_device = std::dynamic_pointer_cast<sycl_device>(_device);
    if (!cl_device)
        throw std::runtime_error("Invalid device type for sycl_engine");
    return cl_device->get_context();
}

const cl::sycl::device& sycl_engine::get_sycl_device() const {
    auto cl_device = std::dynamic_pointer_cast<sycl_device>(_device);
    if (!cl_device)
        throw std::runtime_error("Invalid device type for sycl_engine");
    return cl_device->get_device();
}

cl_device_id sycl_engine::get_ocl_device() const {
    auto cl_device = std::dynamic_pointer_cast<sycl_device>(_device);
    if (!cl_device)
        throw std::runtime_error("Invalid device type for sycl_engine");
    return cl_device->get_ocl_device();
}

memory::ptr sycl_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    if (layout.bytes_count() > get_device_info().max_alloc_mem_size) {
        throw std::runtime_error("exceeded max size of memory object allocation");
    }

    _memory_pool->add_memory_used(layout.bytes_count());

    if (layout.format.is_image_2d()) {
        throw std::runtime_error("Unsupported alloc type");
        // return std::make_shared<sycl::gpu_image2d>(this, layout);
    } else if (type == allocation_type::cl_mem) {
        return std::make_shared<sycl::gpu_buffer>(this, layout);
    } else {
        throw std::runtime_error("Unsupported alloc type");
        // return std::make_shared<sycl::gpu_usm>(this, layout, type);
    }
}

memory::ptr sycl_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    if (memory.get_engine() != this)
        throw std::runtime_error("trying to reinterpret buffer allocated by a different engine");

    if (new_layout.format.is_image() && !memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret non-image buffer as image");

    if (!new_layout.format.is_image() && memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret image buffer as non-image buffer");

    if (new_layout.format.is_image_2d()) {
        throw std::runtime_error("Unsupported alloc type in reinterpret_buffer");
    //    return std::make_shared<sycl::gpu_image2d>(this,
    //                              new_layout,
    //                              reinterpret_cast<const sycl::gpu_image2d&>(memory).get_buffer());
    } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
        throw std::runtime_error("Unsupported alloc type in reinterpret_buffer");
        // return std::make_shared<sycl::gpu_usm>(this,
        //                          new_layout,
        //                          reinterpret_cast<const sycl::gpu_usm&>(memory).get_buffer(),
        //                          memory.get_allocation_type());
    } else {
        return std::make_shared<sycl::gpu_buffer>(this,
                                new_layout,
                                reinterpret_cast<const sycl::gpu_buffer&>(memory).get_buffer());
    }
}

memory::ptr sycl_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    throw std::runtime_error("reinterpret_handle is not implemented");
//    try {
//         if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
//             cl::Image2D img(static_cast<cl_mem>(params.mem), true);
//             return std::make_shared<sycl::gpu_image2d>(this, new_layout, img);
//         } else if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_vasurface) {
//             return std::make_shared<sycl::gpu_media_buffer>(this, new_layout, params);
// #ifdef _WIN32
//         } else if (params.mem_type == shared_mem_type::shared_mem_dxbuffer) {
//             return std::make_shared<sycl::gpu_dx_buffer>(this, new_layout, params);
// #endif
//         } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
//             cl::Buffer buf(static_cast<cl_mem>(params.mem), true);
//             return std::make_shared<sycl::gpu_buffer>(this, new_layout, buf);
//         } else {
//             throw std::runtime_error("unknown shared object fromat or type");
//         }
//     }
//     catch (const cl::Error& clErr) {
//         switch (clErr.err()) {
//         case CL_MEM_OBJECT_ALLOCATION_FAILURE:
//         case CL_OUT_OF_RESOURCES:
//         case CL_OUT_OF_HOST_MEMORY:
//         case CL_INVALID_BUFFER_SIZE:
//             throw std::runtime_error("out of GPU resources");
//         default:
//             throw std::runtime_error("GPU buffer allocation failed");
//         }
//     }
}

bool sycl_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
        return (reinterpret_cast<const sycl::gpu_buffer&>(mem1).get_buffer() ==
                reinterpret_cast<const sycl::gpu_buffer&>(mem2).get_buffer());
    // else
    //     return (reinterpret_cast<const sycl::gpu_usm&>(mem1).get_buffer() ==
    //             reinterpret_cast<const sycl::gpu_usm&>(mem2).get_buffer());

    throw std::runtime_error("Unsupported alloc type in is_the_same_buffer");
}

void* sycl_engine::get_user_context() const {
    auto& cl_device = dynamic_cast<sycl_device&>(*_device);
    return static_cast<void*>(cl_device.get_context().get());
}

bool sycl_engine::extension_supported(std::string extension) const {
    return std::find(_extensions.begin(), _extensions.end(), extension) != _extensions.end();
}

stream::ptr sycl_engine::create_stream() const {
    return std::make_shared<sycl_stream>(*this);
}

stream& sycl_engine::get_program_stream() const {
    return *_program_stream;
}

std::shared_ptr<cldnn::engine> sycl_engine::create(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return std::make_shared<sycl::sycl_engine>(device, runtime_type, configuration);
}

std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return sycl_engine::create(device, runtime_type, configuration);
}

}  // namespace sycl
}  // namespace cldnn
