// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "ze_kernel_builder.hpp"
#include "ze_api.h"
#include "ze_engine_factory.hpp"
#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "ze_stream.hpp"
#include "ze_device.hpp"
#include "ze_kernel.hpp"
#include "ze_module_holder.hpp"
#include "ze_kernel_holder.hpp"
#include <exception>
#include <vector>
#include <memory>
#include <stdexcept>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ze.hpp>
#endif
namespace cldnn {
namespace ze {

ze_engine::ze_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::ze, "[GPU] Invalid runtime type specified for ZE engine. Only ZE runtime is supported");

    auto casted = dynamic_cast<ze_device*>(dev.get());
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type passed to ze engine");

    _service_stream.reset(new ze_stream(*this, ExecutionConfig()));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void ze_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        auto casted = std::dynamic_pointer_cast<ze_device>(_device);
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ze_interop::make_engine(casted->get_driver(), casted->get_device(), casted->get_context()));
    }
}
#endif

const ze_driver_handle_t ze_engine::get_driver() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_driver();
}

const ze_context_handle_t ze_engine::get_context() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_context();
}

const ze_device_handle_t ze_engine::get_device() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_device();
}

allocation_type ze_engine::detect_usm_allocation_type(const void* memory) const {
    return ze::gpu_usm::detect_allocation_type(this, memory);
}

memory::ptr ze_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res = std::make_shared<ze::gpu_usm>(this, layout, type);

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const std::exception& e) {
        OPENVINO_THROW("[GPU] Failed to allocate memory: ", e.what());
    }
}

memory::ptr ze_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            return std::make_shared<ze::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::gpu_usm&>(memory).get_buffer(),
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
    }

    OPENVINO_THROW("[GPU] Trying to reinterpret non usm buffer");
}

memory::ptr ze_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    if (params.mem_type == shared_mem_type::shared_mem_usm) {
        ze::UsmMemory usm_buffer(get_context(), get_device(), params.mem);
        size_t actual_mem_size = 0;
        OV_ZE_EXPECT(zeMemGetAddressRange(get_context(), params.mem, nullptr, &actual_mem_size));
        auto requested_mem_size = new_layout.bytes_count();
        OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared USM buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
        return std::make_shared<ze::gpu_usm>(this, new_layout, usm_buffer, nullptr);
    } else {
        OPENVINO_THROW("[GPU] Unsupported shared memory type: ", params.mem_type);
    }
}

memory_ptr ze_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] Trying to create a subbuffer from a buffer allocated by a different engine");
    if (new_layout.format.is_image_2d()) {
        OPENVINO_NOT_IMPLEMENTED;
    }
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(memory.get_allocation_type()), "[GPU] Trying to create subbuffer for non usm memory");
    auto& new_buf = reinterpret_cast<const ze::gpu_usm&>(memory);
    auto ptr = new_buf.get_buffer().get();
    auto sub_buffer = ze::UsmMemory(get_context(), get_device(), ptr, byte_offset);
    return std::make_shared<ze::gpu_usm>(this,
                             new_layout,
                             sub_buffer,
                             memory.get_allocation_type(),
                             memory.get_mem_tracker());
}

bool ze_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    return (reinterpret_cast<const ze::gpu_usm&>(mem1).get_buffer().get() == reinterpret_cast<const ze::gpu_usm&>(mem2).get_buffer().get());
}

std::shared_ptr<kernel_builder> ze_engine::create_kernel_builder() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return std::make_shared<ze_kernel_builder>(*casted);
}

void* ze_engine::get_user_context() const {
    auto& casted = downcast<ze_device>(*_device);
    return static_cast<void*>(casted.get_context());
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ze_stream>(*this, config);
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<cldnn::engine> ze_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ze::ze_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type) {
    return ze_engine::create(device, runtime_type);
}

}  // namespace ze
}  // namespace cldnn
