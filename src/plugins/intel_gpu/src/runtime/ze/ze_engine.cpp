// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_engine.hpp"
#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "ze_stream.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

namespace cldnn {
namespace ze {

ze_engine::ze_engine(const device::ptr dev, runtime_types runtime_type, const engine_configuration& conf)
    : engine(dev, conf) {
    if (runtime_type != runtime_types::ze) {
        throw std::runtime_error("Invalid runtime type specified for ze engine. Only ze runtime is supported");
    }

    auto casted = dynamic_cast<ze_device*>(dev.get());
    if (!casted)
        throw std::runtime_error("[CLDNN] Invalid device type passed to ze engine");

    _program_stream.reset(new ze_stream(*this));
}

const ze_driver_handle_t ze_engine::get_driver() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    if (!casted)
        throw std::runtime_error("Invalid device type for ze_engine");
    return casted->get_driver();
}

const ze_context_handle_t ze_engine::get_context() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    if (!casted)
        throw std::runtime_error("Invalid device type for ze_engine");
    return casted->get_context();
}

const ze_device_handle_t ze_engine::get_device() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    if (!casted)
        throw std::runtime_error("Invalid device type for ze_engine");
    return casted->get_device();
}

memory::ptr ze_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    if (layout.bytes_count() > get_device_info().max_alloc_mem_size) {
        throw std::runtime_error("exceeded max size of memory object allocation");
    }

    memory::ptr res = std::make_shared<ze::gpu_usm>(this, layout, type);

    if (reset || res->is_memory_reset_needed(layout)) {
        res->fill(get_program_stream());
    }

    return res;
}

memory::ptr ze_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    if (memory.get_engine() != this)
        throw std::runtime_error("trying to reinterpret buffer allocated by a different engine");

    if (new_layout.format.is_image() && !memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret non-image buffer as image");

    if (!new_layout.format.is_image() && memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret image buffer as non-image buffer");

    return nullptr;
    // try {
    //     if (new_layout.format.is_image_2d()) {
    //        return std::make_shared<ze::gpu_image2d>(this,
    //                                  new_layout,
    //                                  reinterpret_cast<const ze::gpu_image2d&>(memory).get_buffer());
    //     } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
    //         return std::make_shared<ze::gpu_usm>(this,
    //                                  new_layout,
    //                                  reinterpret_cast<const ze::gpu_usm&>(memory).get_buffer(),
    //                                  memory.get_allocation_type());
    //     } else {
    //        return std::make_shared<ze::gpu_buffer>(this,
    //                                 new_layout,
    //                                 reinterpret_cast<const ze::gpu_buffer&>(memory).get_buffer());
    //     }
    // } catch (cl::Error const& err) {
    //     throw ze::ze_error(err);
    // }
}

memory::ptr ze_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    return nullptr;
//    try {
//         if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_image) {
//             cl::Image2D img(static_cast<cl_mem>(params.mem), true);
//             return std::make_shared<ze::gpu_image2d>(this, new_layout, img);
//         } else if (new_layout.format.is_image_2d() && params.mem_type == shared_mem_type::shared_mem_vasurface) {
//             return std::make_shared<ze::gpu_media_buffer>(this, new_layout, params);
// #ifdef _WIN32
//         } else if (params.mem_type == shared_mem_type::shared_mem_dxbuffer) {
//             return std::make_shared<ze::gpu_dx_buffer>(this, new_layout, params);
// #endif
//         } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
//             cl::Buffer buf(static_cast<cl_mem>(params.mem), true);
//             return std::make_shared<ze::gpu_buffer>(this, new_layout, buf);
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

bool ze_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    return false;
    // if (mem1.get_engine() != this || mem2.get_engine() != this)
    //     return false;
    // if (mem1.get_allocation_type() != mem2.get_allocation_type())
    //     return false;
    // if (&mem1 == &mem2)
    //     return true;

    // if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
    //     return (reinterpret_cast<const ze::gpu_buffer&>(mem1).get_buffer() ==
    //             reinterpret_cast<const ze::gpu_buffer&>(mem2).get_buffer());
    // else
    //     return (reinterpret_cast<const ze::gpu_usm&>(mem1).get_buffer() ==
    //             reinterpret_cast<const ze::gpu_usm&>(mem2).get_buffer());
}

void* ze_engine::get_user_context() const {
    auto& casted = downcast<ze_device>(*_device);
    return static_cast<void*>(casted.get_driver());
}

stream::ptr ze_engine::create_stream() const {
    //return nullptr;
    return std::make_shared<ze_stream>(*this);
}

stream& ze_engine::get_program_stream() const {
    return *_program_stream;
}

std::shared_ptr<cldnn::engine> ze_engine::create(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return std::make_shared<ze_engine>(device, runtime_type, configuration);
}

std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration) {
    return ze_engine::create(device, runtime_type, configuration);
}

}  // namespace ze
}  // namespace cldnn
