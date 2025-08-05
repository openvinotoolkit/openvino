// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "ze/ze_kernel.hpp"
#include "ze_api.h"
#include "ze_engine_factory.hpp"
#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "ze_stream.hpp"
#include "ze_device.hpp"
#include <exception>
#include <vector>
#include <memory>
#include <stdexcept>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_l0.hpp>
#endif
namespace cldnn {
namespace ze {

namespace {

ze_module_handle_t ze_create_module_with_level_zero(const cldnn::ze::ze_engine& engine, std::vector<uint8_t> binary) {
    auto desc = ze_module_desc_t();
    desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    desc.format = ZE_MODULE_FORMAT_NATIVE;
    desc.inputSize = binary.size();
    desc.pInputModule = binary.data();
    desc.pBuildFlags = "";
    desc.pConstants = nullptr;

    ze_module_handle_t ze_module;

    auto ze_device = engine.get_device();
    auto ze_ctx = engine.get_context();
    zeModuleCreate(ze_ctx, ze_device, &desc, &ze_module, nullptr);
    return ze_module;
}

}  // namespace

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
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::l0_interop::make_engine(casted->get_driver(), casted->get_device(), casted->get_context()));
    }
}

dnnl::engine& ze_engine::get_onednn_engine() const {
    OPENVINO_ASSERT(_onednn_engine, "[GPU] Can't get onednn engine handle as it was not initialized. Please check that create_onednn_engine() was called");
    return *_onednn_engine;
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

bool ze_engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type), "[GPU] Unsupported allocation type: ", type);

    bool exceed_allocatable_mem_size = (layout.bytes_count() > get_device_info().max_alloc_mem_size);

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_allocatable_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

    OPENVINO_ASSERT(!exceed_allocatable_mem_size,
                    "[GPU] Exceeded max size of memory object allocation: ",
                    "requested ", layout.bytes_count(), " bytes, "
                    "but max alloc size supported by device is ", get_device_info().max_alloc_mem_size, " bytes.",
                    "Please try to reduce batch size or use lower precision.");

    auto used_mem = get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    auto exceed_available_mem_size = (layout.bytes_count() + used_mem > get_max_memory_size());

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_available_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

#ifdef __unix__
    // Prevent from being killed by Ooo Killer of Linux
    OPENVINO_ASSERT(!exceed_available_mem_size,
                    "[GPU] Exceeded max size of memory allocation: ",
                    "Required ", layout.bytes_count(), " bytes, already occupied : ", used_mem, " bytes, ",
                    "but available memory size is ", get_max_memory_size(), " bytes");
#else
    if (exceed_available_mem_size) {
        GPU_DEBUG_COUT << "[Warning] [GPU] Exceeded max size of memory allocation: " << "Required " << layout.bytes_count() << " bytes, already occupied : "
                       << used_mem << " bytes, but available memory size is " << get_max_memory_size() << " bytes" << std::endl;
        GPU_DEBUG_COUT << "Please note that performance might drop due to memory swap." << std::endl;
        return false;
    }
#endif

    return true;
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

    return nullptr;
}

memory::ptr ze_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    if (params.mem_type == shared_mem_type::shared_mem_usm) {
        ze::UsmMemory usm_buffer(get_context(), get_device(), params.mem);
        size_t actual_mem_size = 0;
        zeMemGetAddressRange(get_context(), params.mem, nullptr, &actual_mem_size);
        auto requested_mem_size = new_layout.bytes_count();
        OPENVINO_ASSERT(actual_mem_size >= requested_mem_size,
                            "[GPU] shared USM buffer has smaller size (", actual_mem_size,
                            ") than specified layout (", requested_mem_size, ")");
        return std::make_shared<ze::gpu_usm>(this, new_layout, usm_buffer, nullptr);
    } else {
        return nullptr;
    }
}

memory_ptr ze_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] Trying to create a subbuffer from a buffer allocated by a different engine");
    if (new_layout.format.is_image_2d()) {
        OPENVINO_NOT_IMPLEMENTED;
    }
    if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
        auto& new_buf = reinterpret_cast<const ze::gpu_usm&>(memory);
        auto ptr = new_buf.get_buffer().get();
        auto sub_buffer = ze::UsmMemory(get_context(), get_device(), ptr, byte_offset);
        return std::make_shared<ze::gpu_usm>(this,
                                 new_layout,
                                 sub_buffer,
                                 memory.get_allocation_type(),
                                 memory.get_mem_tracker());
    } else {
        OPENVINO_THROW("[GPU] Trying to create subbuffer for non usm memory");
    }
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

kernel::ptr ze_engine::prepare_kernel(const kernel::ptr kernel) const {
    if (std::dynamic_pointer_cast<const ze_kernel>(kernel)) {
        return kernel;
    } else {
        auto binary = kernel->get_binary();
        ze_module_handle_t ze_module = ze_create_module_with_level_zero(*this, binary);
        ze_kernel_handle_t ze_kernel;
        auto entry_point = kernel->get_id();
        ze_kernel_desc_t desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC , nullptr, 0, entry_point.c_str()};
        zeKernelCreate(ze_module, &desc, &ze_kernel);
        return std::make_shared<cldnn::ze::ze_kernel>(ze_kernel, ze_module, entry_point);
    }
}

void* ze_engine::get_user_context() const {
    auto& casted = downcast<ze_device>(*_device);
    return static_cast<void*>(casted.get_driver());
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ze_stream>(*this, config);
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    OPENVINO_NOT_IMPLEMENTED;
}

stream& ze_engine::get_service_stream() const {
    return *_service_stream;
}

std::shared_ptr<cldnn::engine> ze_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ze::ze_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type) {
    return ze_engine::create(device, runtime_type);
}

}  // namespace ze
}  // namespace cldnn
