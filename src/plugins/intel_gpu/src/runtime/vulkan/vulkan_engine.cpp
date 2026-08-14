// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_engine.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "openvino/core/except.hpp"
#include "vulkan_engine_factory.hpp"
#include "vulkan_kernel_builder.hpp"
#include "vulkan_memory.hpp"
#include "vulkan_stream.hpp"

namespace cldnn {
namespace vulkan {

vulkan_engine::vulkan_engine(const device::ptr& device, runtime_types runtime_type) : engine(device) {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU][Vulkan] Invalid runtime type for Vulkan engine");
    const auto vulkan_device = get_vulkan_device_object_impl();
    if (!vulkan_device->is_initialized()) {
        vulkan_device->initialize();
    }
    const auto& info = vulkan_device->get_info();
    const auto alignment = std::max<VkDeviceSize>(info.sub_buffer_base_alignment.value_or(1), 1);
    const auto allocation_count = std::max<uint64_t>(vulkan_device->get_max_memory_allocation_count(), 1);
    auto allocation_count_root = std::max<uint64_t>(static_cast<uint64_t>(std::sqrt(allocation_count)), 1);
    if (allocation_count_root * allocation_count_root < allocation_count) {
        ++allocation_count_root;
    }
    const auto per_allocation_size = std::max<uint64_t>(info.max_global_mem_size / allocation_count_root, 1);
    const auto max_block_size = std::max<uint64_t>(info.max_alloc_mem_size, 1);
    auto preferred_block_size = std::min(per_allocation_size, max_block_size);
    preferred_block_size -= preferred_block_size % alignment;
    if (preferred_block_size == 0) {
        preferred_block_size = std::min<uint64_t>(alignment, max_block_size);
    }
    _memory_allocator = std::make_shared<vulkan_memory_allocator>(*this, alignment, preferred_block_size);
    _service_stream = std::make_unique<vulkan_stream>(*this);
}

std::shared_ptr<vulkan_device> vulkan_engine::get_vulkan_device_object_impl() const {
    const auto result = std::dynamic_pointer_cast<vulkan_device>(_device);
    OPENVINO_ASSERT(result != nullptr, "[GPU][Vulkan] Invalid device type passed to Vulkan engine");
    return result;
}

VkDevice vulkan_engine::get_device_handle() const {
    return get_vulkan_device_object_impl()->get_device();
}

VkPhysicalDevice vulkan_engine::get_physical_device() const {
    return get_vulkan_device_object_impl()->get_physical_device();
}

VkQueue vulkan_engine::get_compute_queue() const {
    return get_vulkan_device_object_impl()->get_compute_queue();
}

uint32_t vulkan_engine::get_compute_queue_family() const {
    return get_vulkan_device_object_impl()->get_compute_queue_family();
}

uint32_t vulkan_engine::get_max_push_constants_size() const {
    return get_vulkan_device_object_impl()->get_max_push_constants_size();
}

std::mutex& vulkan_engine::get_queue_mutex() const {
    return get_vulkan_device_object_impl()->get_queue_mutex();
}

std::shared_ptr<vulkan_device> vulkan_engine::get_vulkan_device_object() const {
    return get_vulkan_device_object_impl();
}

memory_ptr vulkan_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU][Vulkan] Cannot allocate an unbounded dynamic layout");
    OPENVINO_ASSERT(!layout.format.is_image(), "[GPU][Vulkan] Image allocations are not supported");
    OPENVINO_ASSERT(type == allocation_type::vulkan_buffer, "[GPU][Vulkan] Unsupported allocation type: ", type);
    check_allocatable(layout, type);

    auto region = allocate_buffer_region(layout.bytes_count());
    auto* tracking_address = region->get_allocation()->mapped_data;
    if (tracking_address != nullptr) {
        tracking_address = static_cast<unsigned char*>(tracking_address) + region->get_offset();
    } else {
        tracking_address = region.get();
    }
    auto memory_tracker = std::make_shared<MemoryTracker>(this, tracking_address, layout.bytes_count(), allocation_type::vulkan_buffer);
    auto result = std::make_shared<vulkan_buffer>(this, layout, std::move(region), 0, std::move(memory_tracker));
    if (reset || result->is_memory_reset_needed(layout)) {
        result->fill(get_service_stream());
    }
    return result;
}

memory_ptr vulkan_engine::reinterpret_handle(const layout&, shared_mem_params) {
    OPENVINO_THROW("[GPU][Vulkan] External Vulkan memory handles are not supported");
}

memory_ptr vulkan_engine::create_subbuffer(const memory& memory, const layout& layout, size_t byte_offset) {
    const auto* source = dynamic_cast<const vulkan_buffer*>(&memory);
    OPENVINO_ASSERT(source != nullptr && memory.get_engine() == this, "[GPU][Vulkan] Cannot create a subbuffer from memory owned by another backend");
    OPENVINO_ASSERT(byte_offset <= source->size() && layout.bytes_count() <= source->size() - byte_offset,
                    "[GPU][Vulkan] Subbuffer range exceeds its parent buffer");
    return std::make_shared<vulkan_buffer>(this, layout, source->get_region(), source->get_view_offset() + byte_offset, source->get_mem_tracker());
}

memory_ptr vulkan_engine::create_hostbuffer(void*, size_t, allocation_type, const layout) {
    OPENVINO_THROW("[GPU][Vulkan] Zero-copy wrapping of host memory is not supported");
}

memory_ptr vulkan_engine::create_hostbuffer(const void*, size_t, allocation_type, const layout) {
    OPENVINO_THROW("[GPU][Vulkan] Zero-copy wrapping of host memory is not supported");
}

memory_ptr vulkan_engine::reinterpret_buffer(const memory& memory, const layout& layout) {
    const auto* source = dynamic_cast<const vulkan_buffer*>(&memory);
    OPENVINO_ASSERT(source != nullptr && memory.get_engine() == this, "[GPU][Vulkan] Cannot reinterpret memory owned by another backend");
    OPENVINO_ASSERT(
        source->get_view_offset() <= source->get_region()->get_size() && layout.bytes_count() <= source->get_region()->get_size() - source->get_view_offset(),
        "[GPU][Vulkan] Reinterpreted layout exceeds the underlying allocation");
    auto result = std::make_shared<vulkan_buffer>(this, layout, source->get_region(), source->get_view_offset(), source->get_mem_tracker());
    result->from_memory_pool = memory.from_memory_pool;
    return result;
}

memory_ptr vulkan_engine::import_buffer(const layout&, ov::intel_gpu::os_handle_param) {
    OPENVINO_THROW("[GPU][Vulkan] External OS memory handle import is not supported");
}

bool vulkan_engine::is_the_same_buffer(const memory& lhs, const memory& rhs) {
    const auto* lhs_buffer = dynamic_cast<const vulkan_buffer*>(&lhs);
    const auto* rhs_buffer = dynamic_cast<const vulkan_buffer*>(&rhs);
    return lhs_buffer != nullptr && rhs_buffer != nullptr && lhs.get_engine() == this && rhs.get_engine() == this &&
           lhs_buffer->get_allocation() == rhs_buffer->get_allocation() && lhs_buffer->get_offset() == rhs_buffer->get_offset();
}

std::shared_ptr<vulkan_buffer_region> vulkan_engine::allocate_buffer_region(size_t size) {
    OPENVINO_ASSERT(_memory_allocator != nullptr, "[GPU][Vulkan] Memory allocator is not initialized");
    return _memory_allocator->allocate(size);
}

vulkan_memory_allocator_stats vulkan_engine::get_memory_allocator_stats() const {
    OPENVINO_ASSERT(_memory_allocator != nullptr, "[GPU][Vulkan] Memory allocator is not initialized");
    return _memory_allocator->get_stats();
}

void* vulkan_engine::get_user_context(runtime_types runtime_type) const {
    OPENVINO_ASSERT(runtime_type == runtime_types::vulkan, "[GPU][Vulkan] Cannot provide a context for runtime ", runtime_type);
    return reinterpret_cast<void*>(get_device_handle());
}

stream_ptr vulkan_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<vulkan_stream>(*this, config);
}

stream_ptr vulkan_engine::create_stream(const ExecutionConfig&, void*) const {
    OPENVINO_THROW("[GPU][Vulkan] External Vulkan queues are not supported");
}

std::shared_ptr<kernel_builder> vulkan_engine::create_kernel_builder() const {
    return std::make_shared<vulkan_kernel_builder>(*this);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void vulkan_engine::create_onednn_engine(const ExecutionConfig&) {
    OPENVINO_THROW("[GPU][Vulkan] oneDNN interop is not supported");
}
#endif

std::shared_ptr<cldnn::engine> create_vulkan_engine(const device::ptr& device, runtime_types runtime_type) {
    return std::make_shared<vulkan_engine>(device, runtime_type);
}

}  // namespace vulkan
}  // namespace cldnn
