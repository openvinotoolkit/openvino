// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_stream.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_event.hpp"
#include "vulkan_kernel.hpp"
#include "vulkan_memory.hpp"

namespace cldnn {
namespace vulkan {
namespace {

class empty_surfaces_lock final : public surfaces_lock {};

bool stream_diagnostics_enabled() {
    const auto* value = std::getenv("OV_GPU_VULKAN_STREAM_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

memory::cptr get_argument_memory(const argument_desc& descriptor, const kernel_arguments_data& data) {
    switch (descriptor.t) {
    case argument_desc::Types::INPUT:
        return data.inputs.at(descriptor.index);
    case argument_desc::Types::OUTPUT:
        return data.outputs.at(descriptor.index);
    case argument_desc::Types::WEIGHTS:
        return data.weights;
    case argument_desc::Types::BIAS:
        return data.bias;
    case argument_desc::Types::SCALE_TABLE:
        return data.scale_table;
    case argument_desc::Types::SLOPE:
        return data.slope;
    case argument_desc::Types::INTERNAL_BUFFER:
        return data.intermediates.at(descriptor.index);
    case argument_desc::Types::CELL:
        return data.cell;
    case argument_desc::Types::WEIGHTS_ZERO_POINTS:
        return data.weights_zero_points;
    case argument_desc::Types::ACTIVATIONS_ZERO_POINTS:
        return data.activations_zero_points;
    case argument_desc::Types::COMPENSATION:
        return data.compensation;
    case argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE:
        return data.fused_op_inputs.at(descriptor.index);
    case argument_desc::Types::SHAPE_INFO:
        return data.shape_info;
    case argument_desc::Types::SCALAR:
    case argument_desc::Types::LOCAL_MEMORY_SIZE:
        return nullptr;
    }
    OPENVINO_THROW("[GPU][Vulkan] Unknown kernel argument type");
}

std::vector<uint8_t> pack_push_constants(const scalars_desc& scalars) {
    std::vector<uint8_t> result;
    result.reserve(scalars.size() * sizeof(uint32_t));
    for (const auto& scalar : scalars) {
        OPENVINO_ASSERT(scalar.t == scalar_desc::Types::UINT32 || scalar.t == scalar_desc::Types::INT32 || scalar.t == scalar_desc::Types::FLOAT32,
                        "[GPU][Vulkan] Only 32-bit scalar push constants are currently supported");
        const auto* bytes = reinterpret_cast<const uint8_t*>(&scalar.v.u32);
        result.insert(result.end(), bytes, bytes + sizeof(uint32_t));
    }
    return result;
}

struct prepared_arguments {
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<vulkan_buffer_allocation::ptr> allocations;
    std::vector<bool> shader_writes;
    std::vector<bool> host_outputs;
    std::vector<memory::cptr> memories;
    std::vector<uint8_t> push_constants;
};

bool is_shader_write_argument(argument_desc::Types type) {
    return type == argument_desc::Types::OUTPUT || type == argument_desc::Types::INTERNAL_BUFFER;
}

prepared_arguments prepare_arguments(const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    prepared_arguments prepared;
    for (const auto& argument : descriptor.arguments) {
        auto memory = get_argument_memory(argument, data);
        if (memory == nullptr) {
            continue;
        }

        const auto* buffer = dynamic_cast<const vulkan_buffer*>(memory.get());
        OPENVINO_ASSERT(buffer != nullptr, "[GPU][Vulkan] Kernel argument is not backed by a Vulkan buffer");
        OPENVINO_ASSERT(buffer->size() > 0, "[GPU][Vulkan] Zero-sized storage buffer arguments are not supported");
        prepared.buffer_infos.push_back({buffer->get_buffer(), buffer->get_offset(), buffer->size()});
        prepared.allocations.push_back(buffer->get_allocation());
        prepared.shader_writes.push_back(is_shader_write_argument(argument.t));
        prepared.host_outputs.push_back(argument.t == argument_desc::Types::OUTPUT);
        prepared.memories.push_back(std::move(memory));
    }
    prepared.push_constants = pack_push_constants(descriptor.scalars);
    return prepared;
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] ", description, " exceeds the 32-bit Vulkan range");
    return static_cast<uint32_t>(value);
}

std::vector<VkBufferMemoryBarrier> make_shader_input_barriers(const prepared_arguments& prepared) {
    std::vector<VkBufferMemoryBarrier> barriers(prepared.buffer_infos.size());
    for (size_t index = 0; index < barriers.size(); ++index) {
        auto& barrier = barriers[index];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | (prepared.shader_writes[index] ? VK_ACCESS_SHADER_WRITE_BIT : 0);
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = prepared.buffer_infos[index].buffer;
        barrier.offset = prepared.buffer_infos[index].offset;
        barrier.size = prepared.buffer_infos[index].range;
    }
    return barriers;
}

std::vector<VkBufferMemoryBarrier> make_host_output_barriers(const prepared_arguments& prepared) {
    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(prepared.buffer_infos.size());
    for (size_t index = 0; index < prepared.buffer_infos.size(); ++index) {
        if (!prepared.host_outputs[index]) {
            continue;
        }

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = prepared.buffer_infos[index].buffer;
        barrier.offset = prepared.buffer_infos[index].offset;
        barrier.size = prepared.buffer_infos[index].range;
        barriers.push_back(barrier);
    }
    return barriers;
}

}  // namespace

struct vulkan_stream::resource_state {
    // These values bound retained resources. They are not
    // selected operating points; the batch limit is chosen per workload below.
    static constexpr size_t max_in_flight_submissions = 8;
    static constexpr size_t max_cached_descriptor_sets = 256;
    static constexpr size_t max_retained_dispatches_per_batch = 8;

    struct descriptor_key {
        std::vector<std::weak_ptr<vulkan_buffer_allocation>> allocations;
        std::vector<VkDescriptorBufferInfo> buffer_infos;

        bool operator<(const descriptor_key& other) const {
            if (allocations.size() != other.allocations.size()) {
                return allocations.size() < other.allocations.size();
            }
            const std::owner_less<std::weak_ptr<vulkan_buffer_allocation>> allocation_less;
            for (size_t index = 0; index < allocations.size(); ++index) {
                if (allocation_less(allocations[index], other.allocations[index])) {
                    return true;
                }
                if (allocation_less(other.allocations[index], allocations[index])) {
                    return false;
                }
                const auto& info = buffer_infos[index];
                const auto& other_info = other.buffer_infos[index];
                if (info.offset != other_info.offset) {
                    return info.offset < other_info.offset;
                }
                if (info.range != other_info.range) {
                    return info.range < other_info.range;
                }
            }
            return false;
        }
    };

    struct descriptor_pool_block {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        uint32_t capacity = 0;
        uint32_t allocated_sets = 0;
    };

    struct slot {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        uint32_t descriptor_capacity = 0;
        uint32_t descriptor_binding_count = 0;
        VkFence fence = VK_NULL_HANDLE;
        std::shared_ptr<vulkan_submission_state> submission;
        std::vector<memory::cptr> retained_memories;
        std::vector<std::weak_ptr<vulkan_buffer_allocation>> descriptor_allocations;
        std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
        std::vector<std::shared_ptr<const void>> retained_kernels;
        size_t recorded_dispatches = 0;
    };

    explicit resource_state(const vulkan_engine& engine)
        : device(engine.get_device_handle()),
          queue(engine.get_compute_queue()),
          queue_mutex(engine.get_queue_mutex()),
          max_work_group_invocations(engine.get_device_info().max_work_group_size),
          subgroup_size(engine.get_device_info().supported_simd_sizes.empty() ? 0 : engine.get_device_info().supported_simd_sizes.front()),
          diagnostics_enabled(stream_diagnostics_enabled()) {
        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = engine.get_compute_queue_family();
        check_vk_result(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");
        slots.reserve(max_in_flight_submissions);
    }

    ~resource_state() {
        finish();
        for (auto& slot : slots) {
            recycle(slot);
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, slot.fence, nullptr);
            }
            if (slot.descriptor_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(device, slot.descriptor_pool, nullptr);
            }
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
        }
        for (auto& [descriptor_count, blocks] : descriptor_cache_pools) {
            for (auto& block : blocks) {
                if (block.pool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(device, block.pool, nullptr);
                }
            }
        }
        if (diagnostics_enabled) {
            std::clog << "[GPU][Vulkan][Stream] descriptor_allocations=" << descriptor_allocations << " descriptor_updates=" << descriptor_updates
                      << " descriptor_reuses=" << descriptor_reuses << " dispatches=" << dispatches << " queue_submissions=" << queue_submissions
                      << " max_batch_size=" << largest_batch << " selected_batch_limit_min=" << (dispatches == 0 ? 0 : smallest_selected_batch_limit)
                      << " selected_batch_limit_max=" << largest_selected_batch_limit << std::endl;
        }
    }

    slot& get_or_begin_batch(uint32_t descriptor_count, size_t local_invocations) {
        const auto selected_limit = select_batch_limit(local_invocations);
        smallest_selected_batch_limit = std::min(smallest_selected_batch_limit, selected_limit);
        largest_selected_batch_limit = std::max(largest_selected_batch_limit, selected_limit);
        if (recording_slot != nullptr) {
            active_batch_limit = std::min(active_batch_limit, selected_limit);
            if (recording_slot->recorded_dispatches < active_batch_limit) {
                return *recording_slot;
            }
            flush();
        }

        auto& slot = acquire(descriptor_count);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk_result(vkBeginCommandBuffer(slot.command_buffer, &begin_info), "vkBeginCommandBuffer");
        recording_slot = &slot;
        active_batch_limit = selected_limit;
        return slot;
    }

    bool descriptor_batch_boundary_required(const prepared_arguments& prepared) const {
        return descriptor_cache.size() >= max_cached_descriptor_sets && descriptor_cache.find(make_descriptor_key(prepared)) == descriptor_cache.end();
    }

    void retain_dispatch(slot& slot, const prepared_arguments& prepared, std::shared_ptr<const void> kernel_lifetime) {
        OPENVINO_ASSERT(recording_slot == &slot && slot.submission == nullptr, "[GPU][Vulkan] Dispatch resources do not belong to the active command batch");
        slot.retained_memories.insert(slot.retained_memories.end(), prepared.memories.begin(), prepared.memories.end());
        slot.retained_kernels.push_back(std::move(kernel_lifetime));
        ++slot.recorded_dispatches;
        ++dispatches;
    }

    bool batch_is_full() const {
        return recording_slot != nullptr && recording_slot->recorded_dispatches >= active_batch_limit;
    }

    std::shared_ptr<vulkan_submission_state> flush() {
        if (recording_slot == nullptr) {
            return latest_submission.lock();
        }

        auto& slot = *recording_slot;
        OPENVINO_ASSERT(slot.recorded_dispatches > 0, "[GPU][Vulkan] Cannot submit an empty command batch");
        check_vk_result(vkEndCommandBuffer(slot.command_buffer), "vkEndCommandBuffer");

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &slot.command_buffer;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            check_vk_result(vkQueueSubmit(queue, 1, &submit_info, slot.fence), "vkQueueSubmit");
        }

        ++queue_submissions;
        largest_batch = std::max(largest_batch, slot.recorded_dispatches);
        recording_slot = nullptr;
        active_batch_limit = 0;
        return mark_submitted(slot);
    }

    void finish() {
        flush();
        for (auto& slot : slots) {
            if (slot.submission != nullptr) {
                slot.submission->wait();
            }
        }
    }

    VkDescriptorSet get_or_update_descriptor_set(slot& slot, const vulkan_pipeline_state& pipeline, const prepared_arguments& prepared) {
        const auto descriptor_count = checked_u32(prepared.buffer_infos.size(), "descriptor count");
        OPENVINO_ASSERT(descriptor_count == pipeline.descriptor_count && slot.descriptor_pool != VK_NULL_HANDLE,
                        "[GPU][Vulkan] Descriptor resources do not match the selected pipeline");

        auto key = make_descriptor_key(prepared);
        const auto cached = descriptor_cache.find(key);
        if (cached != descriptor_cache.end()) {
            ++descriptor_reuses;
            return cached->second;
        }
        if (descriptor_cache.size() < max_cached_descriptor_sets) {
            const auto descriptor_set = allocate_cached_descriptor_set(pipeline.descriptor_set_layout, descriptor_count);
            update_descriptor_set(descriptor_set, prepared.buffer_infos);
            descriptor_cache.emplace(std::move(key), descriptor_set);
            ++descriptor_allocations;
            ++descriptor_updates;
            return descriptor_set;
        }

        // All Vulkan compute pipelines describe consecutive storage-buffer bindings,
        // so layouts with the same binding count are descriptor-set compatible. A
        // cache miss at capacity starts a new batch before this mutable set is used.
        if (slot.descriptor_set == VK_NULL_HANDLE || slot.descriptor_binding_count != descriptor_count) {
            if (slot.descriptor_set != VK_NULL_HANDLE) {
                check_vk_result(vkResetDescriptorPool(device, slot.descriptor_pool, 0), "vkResetDescriptorPool");
            }

            VkDescriptorSetAllocateInfo descriptor_set_info{};
            descriptor_set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptor_set_info.descriptorPool = slot.descriptor_pool;
            descriptor_set_info.descriptorSetCount = 1;
            descriptor_set_info.pSetLayouts = &pipeline.descriptor_set_layout;
            check_vk_result(vkAllocateDescriptorSets(device, &descriptor_set_info, &slot.descriptor_set), "vkAllocateDescriptorSets");
            slot.descriptor_binding_count = descriptor_count;
            slot.descriptor_allocations.clear();
            slot.descriptor_buffer_infos.clear();
            ++descriptor_allocations;
        }

        bool bindings_unchanged =
            slot.descriptor_allocations.size() == prepared.allocations.size() && slot.descriptor_buffer_infos.size() == prepared.buffer_infos.size();
        const std::owner_less<std::weak_ptr<vulkan_buffer_allocation>> allocation_less;
        for (size_t index = 0; bindings_unchanged && index < prepared.buffer_infos.size(); ++index) {
            const auto& cached = slot.descriptor_buffer_infos[index];
            const auto& current = prepared.buffer_infos[index];
            const std::weak_ptr<vulkan_buffer_allocation> current_allocation = prepared.allocations[index];
            const auto& cached_allocation = slot.descriptor_allocations[index];
            bindings_unchanged = !cached_allocation.expired() && !allocation_less(cached_allocation, current_allocation) &&
                                 !allocation_less(current_allocation, cached_allocation) && cached.buffer == current.buffer &&
                                 cached.offset == current.offset && cached.range == current.range;
        }
        if (bindings_unchanged) {
            ++descriptor_reuses;
            return slot.descriptor_set;
        }

        update_descriptor_set(slot.descriptor_set, prepared.buffer_infos);
        slot.descriptor_allocations.clear();
        slot.descriptor_allocations.reserve(prepared.allocations.size());
        for (const auto& allocation : prepared.allocations) {
            slot.descriptor_allocations.emplace_back(allocation);
        }
        slot.descriptor_buffer_infos = prepared.buffer_infos;
        ++descriptor_updates;
        return slot.descriptor_set;
    }

    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex& queue_mutex;
    uint64_t max_work_group_invocations = 0;
    uint64_t subgroup_size = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::vector<slot> slots;
    size_t next_slot = 0;
    bool diagnostics_enabled = false;
    uint64_t descriptor_allocations = 0;
    uint64_t descriptor_updates = 0;
    uint64_t descriptor_reuses = 0;
    uint64_t dispatches = 0;
    uint64_t queue_submissions = 0;
    size_t largest_batch = 0;
    size_t smallest_selected_batch_limit = max_retained_dispatches_per_batch;
    size_t largest_selected_batch_limit = 0;
    std::map<descriptor_key, VkDescriptorSet> descriptor_cache;
    std::map<uint32_t, std::vector<descriptor_pool_block>> descriptor_cache_pools;

private:
    size_t select_batch_limit(size_t local_invocations) const {
        const auto usable_local_invocations = std::max<size_t>(local_invocations, 1);
        const auto work_group_capacity = std::max<uint64_t>(max_work_group_invocations / usable_local_invocations, 1);
        if (subgroup_size == 0) {
            return 1;
        }
        const auto subgroup_capacity = std::max<uint64_t>(usable_local_invocations / subgroup_size, 1);
        return std::min<size_t>({work_group_capacity, subgroup_capacity, max_retained_dispatches_per_batch});
    }

    static descriptor_key make_descriptor_key(const prepared_arguments& prepared) {
        descriptor_key key;
        key.buffer_infos = prepared.buffer_infos;
        key.allocations.reserve(prepared.allocations.size());
        for (const auto& allocation : prepared.allocations) {
            key.allocations.emplace_back(allocation);
        }
        return key;
    }

    slot& acquire(uint32_t descriptor_count) {
        for (auto& slot : slots) {
            if (slot.submission == nullptr) {
                ensure_descriptor_pool(slot, descriptor_count);
                return slot;
            }
            if (slot.submission->is_complete()) {
                recycle(slot);
                ensure_descriptor_pool(slot, descriptor_count);
                return slot;
            }
        }

        if (slots.size() < max_in_flight_submissions) {
            slots.emplace_back();
            auto& slot = slots.back();

            VkCommandBufferAllocateInfo command_buffer_info{};
            command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            command_buffer_info.commandPool = command_pool;
            command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            command_buffer_info.commandBufferCount = 1;
            check_vk_result(vkAllocateCommandBuffers(device, &command_buffer_info, &slot.command_buffer), "vkAllocateCommandBuffers");

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check_vk_result(vkCreateFence(device, &fence_info, nullptr, &slot.fence), "vkCreateFence");

            ensure_descriptor_pool(slot, descriptor_count);
            return slot;
        }

        auto& slot = slots[next_slot++ % slots.size()];
        slot.submission->wait();
        recycle(slot);
        ensure_descriptor_pool(slot, descriptor_count);
        return slot;
    }

    std::shared_ptr<vulkan_submission_state> mark_submitted(slot& slot) {
        OPENVINO_ASSERT(slot.fence != VK_NULL_HANDLE && slot.submission == nullptr, "[GPU][Vulkan] Submission resources are already in use");
        slot.submission = std::make_shared<vulkan_submission_state>(device, queue, slot.fence);
        slot.fence = VK_NULL_HANDLE;
        latest_submission = slot.submission;
        return slot.submission;
    }
    VkDescriptorSet allocate_cached_descriptor_set(VkDescriptorSetLayout layout, uint32_t descriptor_count) {
        auto& blocks = descriptor_cache_pools[descriptor_count];
        if (blocks.empty() || blocks.back().allocated_sets == blocks.back().capacity) {
            const auto remaining_cache_entries = max_cached_descriptor_sets - descriptor_cache.size();
            const auto descriptors_per_set = std::max<uint32_t>(descriptor_count, 1);
            const auto pool_capacity = checked_u32(std::max<size_t>(remaining_cache_entries / descriptors_per_set, 1), "descriptor pool capacity");

            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_size.descriptorCount = checked_u32(static_cast<size_t>(descriptor_count) * pool_capacity, "descriptor pool storage-buffer count");

            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.maxSets = pool_capacity;
            pool_info.poolSizeCount = descriptor_count == 0 ? 0 : 1;
            pool_info.pPoolSizes = descriptor_count == 0 ? nullptr : &pool_size;

            descriptor_pool_block block;
            block.capacity = pool_capacity;
            check_vk_result(vkCreateDescriptorPool(device, &pool_info, nullptr, &block.pool), "vkCreateDescriptorPool(cache)");
            blocks.push_back(block);
        }

        auto& block = blocks.back();
        VkDescriptorSetAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate_info.descriptorPool = block.pool;
        allocate_info.descriptorSetCount = 1;
        allocate_info.pSetLayouts = &layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        check_vk_result(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set), "vkAllocateDescriptorSets(cache)");
        ++block.allocated_sets;
        return descriptor_set;
    }

    void update_descriptor_set(VkDescriptorSet descriptor_set, const std::vector<VkDescriptorBufferInfo>& buffer_infos) {
        std::vector<VkWriteDescriptorSet> descriptor_writes(buffer_infos.size());
        for (uint32_t index = 0; index < descriptor_writes.size(); ++index) {
            descriptor_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[index].dstSet = descriptor_set;
            descriptor_writes[index].dstBinding = index;
            descriptor_writes[index].descriptorCount = 1;
            descriptor_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_writes[index].pBufferInfo = &buffer_infos[index];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
    }

    void recycle(slot& slot) {
        if (slot.submission == nullptr) {
            return;
        }

        slot.submission->wait();
        slot.fence = slot.submission->release_fence();
        slot.submission.reset();
        slot.retained_memories.clear();
        slot.retained_kernels.clear();
        slot.recorded_dispatches = 0;
        check_vk_result(vkResetFences(device, 1, &slot.fence), "vkResetFences");
        check_vk_result(vkResetCommandBuffer(slot.command_buffer, 0), "vkResetCommandBuffer");
        // Keep the descriptor set allocated. The next dispatch can reuse both
        // the set and its bindings after this slot's fence has completed.
    }

    void ensure_descriptor_pool(slot& slot, uint32_t descriptor_count) {
        if (slot.descriptor_pool != VK_NULL_HANDLE && slot.descriptor_capacity >= descriptor_count) {
            return;
        }
        if (slot.descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, slot.descriptor_pool, nullptr);
            slot.descriptor_pool = VK_NULL_HANDLE;
        }
        slot.descriptor_set = VK_NULL_HANDLE;
        slot.descriptor_binding_count = 0;
        slot.descriptor_allocations.clear();
        slot.descriptor_buffer_infos.clear();

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = descriptor_count;

        VkDescriptorPoolCreateInfo descriptor_pool_info{};
        descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_info.maxSets = 1;
        descriptor_pool_info.poolSizeCount = descriptor_count == 0 ? 0 : 1;
        descriptor_pool_info.pPoolSizes = descriptor_count == 0 ? nullptr : &pool_size;
        check_vk_result(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &slot.descriptor_pool), "vkCreateDescriptorPool");
        slot.descriptor_capacity = descriptor_count;
    }

    slot* recording_slot = nullptr;
    size_t active_batch_limit = 0;
    std::weak_ptr<vulkan_submission_state> latest_submission;
};

vulkan_stream::vulkan_stream(const vulkan_engine& engine)
    : stream(QueueTypes::in_order, SyncMethods::none),
      _engine(engine),
      _resources(std::make_unique<resource_state>(engine)) {}

vulkan_stream::vulkan_stream(const vulkan_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config)),
      _engine(engine),
      _resources(std::make_unique<resource_state>(engine)) {}

vulkan_stream::~vulkan_stream() = default;

void vulkan_stream::flush() const {
    _resources->flush();
}

void vulkan_stream::finish() const {
    _resources->finish();
}

void vulkan_stream::wait() {
    finish();
}

void vulkan_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& descriptor, const kernel_arguments_data& data) {
    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot bind arguments to a kernel from another backend");
    const auto prepared = prepare_arguments(descriptor, data);
    const auto specialized_local_size_x = descriptor.specialize_local_size_x ? static_cast<uint32_t>(descriptor.workGroups.local.at(0)) : 0U;
    vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()),
                                      static_cast<uint32_t>(prepared.push_constants.size()),
                                      specialized_local_size_x,
                                      descriptor.specialization_constants);
}

event::ptr vulkan_stream::enqueue_kernel(kernel& kernel,
                                         const kernel_arguments_desc& descriptor,
                                         const kernel_arguments_data& data,
                                         const std::vector<event::ptr>& dependencies,
                                         bool is_output_event) {
    for (const auto& dependency : dependencies) {
        if (dependency == nullptr) {
            continue;
        }
        const auto* vk_event = dynamic_cast<const vulkan_event*>(dependency.get());
        if (vk_event == nullptr || !vk_event->is_device_submission(_engine.get_device_handle(), _engine.get_compute_queue())) {
            _resources->flush();
            dependency->wait();
        }
    }

    auto* vk_kernel = dynamic_cast<vulkan_kernel*>(&kernel);
    OPENVINO_ASSERT(vk_kernel != nullptr, "[GPU][Vulkan] Cannot dispatch a kernel from another backend");
    const auto prepared = prepare_arguments(descriptor, data);
    const auto specialized_local_size_x = descriptor.specialize_local_size_x ? static_cast<uint32_t>(descriptor.workGroups.local.at(0)) : 0U;
    const auto pipeline = vk_kernel->get_or_create_pipeline(static_cast<uint32_t>(prepared.buffer_infos.size()),
                                                            static_cast<uint32_t>(prepared.push_constants.size()),
                                                            specialized_local_size_x,
                                                            descriptor.specialization_constants);

    OPENVINO_ASSERT(descriptor.workGroups.global.size() == 3 && descriptor.workGroups.local.size() == 3,
                    "[GPU][Vulkan] Compute dispatch requires three-dimensional global and local work-group sizes");
    uint32_t group_counts[3]{};
    size_t local_invocations = 1;
    for (size_t axis = 0; axis < 3; ++axis) {
        const auto local_size = descriptor.workGroups.local[axis];
        OPENVINO_ASSERT(local_size > 0, "[GPU][Vulkan] Local work-group size cannot be zero");
        OPENVINO_ASSERT(local_invocations <= std::numeric_limits<size_t>::max() / local_size,
                        "[GPU][Vulkan] Local work-group invocation count overflows size_t");
        local_invocations *= local_size;
        group_counts[axis] = static_cast<uint32_t>((descriptor.workGroups.global[axis] + local_size - 1) / local_size);
    }

    if (_resources->descriptor_batch_boundary_required(prepared)) {
        _resources->flush();
    }
    auto& resources = _resources->get_or_begin_batch(checked_u32(prepared.buffer_infos.size(), "descriptor count"), local_invocations);

    const auto descriptor_set = _resources->get_or_update_descriptor_set(resources, *pipeline, prepared);

    const auto shader_input_barriers = make_shader_input_barriers(prepared);
    vkCmdPipelineBarrier(resources.command_buffer,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         checked_u32(shader_input_barriers.size(), "input barrier count"),
                         shader_input_barriers.data(),
                         0,
                         nullptr);

    vkCmdBindPipeline(resources.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(resources.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    if (!prepared.push_constants.empty()) {
        vkCmdPushConstants(resources.command_buffer,
                           pipeline->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           static_cast<uint32_t>(prepared.push_constants.size()),
                           prepared.push_constants.data());
    }

    vkCmdDispatch(resources.command_buffer, group_counts[0], group_counts[1], group_counts[2]);

    if (is_output_event) {
        const auto host_output_barriers = make_host_output_barriers(prepared);
        if (!host_output_barriers.empty()) {
            vkCmdPipelineBarrier(resources.command_buffer,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT,
                                 0,
                                 0,
                                 nullptr,
                                 checked_u32(host_output_barriers.size(), "output barrier count"),
                                 host_output_barriers.data(),
                                 0,
                                 nullptr);
        }
    }
    _resources->retain_dispatch(resources, prepared, pipeline);
    if (is_output_event || m_sync_method == SyncMethods::events || _resources->batch_is_full()) {
        return std::make_shared<vulkan_event>(_resources->flush());
    }
    return nullptr;
}

event::ptr vulkan_stream::enqueue_marker(const std::vector<event::ptr>& dependencies, bool) {
    _resources->flush();
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::enqueue_barrier() {
    finish();
}

event::ptr vulkan_stream::group_events(const std::vector<event::ptr>& dependencies) {
    wait_for_events(dependencies);
    return create_user_event(true);
}

void vulkan_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (!events.empty()) {
        _resources->flush();
    }
    for (const auto& event : events) {
        if (event != nullptr) {
            event->wait();
        }
    }
}

event::ptr vulkan_stream::create_user_event(bool set) {
    return std::make_shared<vulkan_event>(set);
}

event::ptr vulkan_stream::create_base_event() {
    return std::make_shared<vulkan_event>();
}

std::unique_ptr<surfaces_lock> vulkan_stream::create_surfaces_lock(const std::vector<memory::ptr>&) const {
    return std::make_unique<empty_surfaces_lock>();
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& vulkan_stream::get_onednn_stream() {
    OPENVINO_THROW("[GPU][Vulkan] oneDNN stream interop is not supported");
}
#endif

}  // namespace vulkan
}  // namespace cldnn
