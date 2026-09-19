// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_command_sequence_cache.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_descriptor_cache.hpp"
#include "vulkan_kernel.hpp"

namespace cldnn::vulkan {
namespace {

void check_vk_result(VkResult result, const char* operation) {
    OPENVINO_ASSERT(result == VK_SUCCESS, "[GPU][Vulkan] ", operation, " failed with VkResult ", static_cast<int>(result));
}

uint32_t checked_u32(size_t value, const char* description) {
    OPENVINO_ASSERT(value <= std::numeric_limits<uint32_t>::max(), "[GPU][Vulkan] ", description, " exceeds the 32-bit Vulkan range");
    return static_cast<uint32_t>(value);
}

}  // namespace

void record_vulkan_synchronization(VkCommandBuffer command_buffer,
                                   bool external_dependency_barrier,
                                   const VkBufferMemoryBarrier2* barriers,
                                   size_t barrier_count) {
    if (external_dependency_barrier) {
        VkMemoryBarrier2 memory_barrier{};
        memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory_barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        memory_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        memory_barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

        VkDependencyInfo dependency_info{};
        dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency_info.memoryBarrierCount = 1;
        dependency_info.pMemoryBarriers = &memory_barrier;
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }
    if (barrier_count > 0) {
        VkDependencyInfo dependency_info{};
        dependency_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency_info.bufferMemoryBarrierCount = checked_u32(barrier_count, "buffer hazard barrier count");
        dependency_info.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(command_buffer, &dependency_info);
    }
}

void record_vulkan_dispatch(VkCommandBuffer command_buffer,
                            const vulkan_pipeline_state& pipeline,
                            VkDescriptorSet descriptor_set,
                            const std::vector<uint8_t>& push_constants,
                            const std::array<uint32_t, 3>& group_counts) {
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    if (!push_constants.empty()) {
        vkCmdPushConstants(command_buffer,
                           pipeline.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           checked_u32(push_constants.size(), "push constant size"),
                           push_constants.data());
    }
    vkCmdDispatch(command_buffer, group_counts[0], group_counts[1], group_counts[2]);
}

vulkan_command_sequence_cache::vulkan_command_sequence_cache(VkDevice device, uint32_t queue_family_index) : _device(device) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE, "[GPU][Vulkan] Command sequence cache requires a valid Vulkan device");
    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = queue_family_index;
    check_vk_result(vkCreateCommandPool(_device, &command_pool_info, nullptr, &_command_pool), "vkCreateCommandPool(replay)");
}

vulkan_command_sequence_cache::~vulkan_command_sequence_cache() {
    if (_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(_device, _command_pool, nullptr);
    }
}

void vulkan_command_sequence_cache::begin_batch() {
    _current_dispatches.clear();
    _current_barriers.clear();
    _current_allocations.clear();
    _current_cacheable = true;
}

void vulkan_command_sequence_cache::append(std::shared_ptr<const vulkan_pipeline_state> pipeline,
                                           VkDescriptorSet descriptor_set,
                                           std::vector<uint8_t>&& push_constants,
                                           const std::array<uint32_t, 3>& group_counts,
                                           const std::vector<VkBufferMemoryBarrier2>& barriers,
                                           bool external_dependency_barrier,
                                           const std::vector<vulkan_buffer_allocation::ptr>& allocations,
                                           bool descriptor_is_immutable) {
    const auto barrier_offset = _current_barriers.size();
    _current_barriers.insert(_current_barriers.end(), barriers.begin(), barriers.end());
    _current_dispatches.push_back(
        {std::move(pipeline), descriptor_set, group_counts, std::move(push_constants), barrier_offset, barriers.size(), external_dependency_barrier});
    _current_allocations.insert(_current_allocations.end(), allocations.begin(), allocations.end());
    _current_cacheable = _current_cacheable && descriptor_is_immutable;
}

vulkan_command_sequence_cache::resolution vulkan_command_sequence_cache::resolve(VkCommandBuffer transient_command_buffer, uint64_t submission_sequence) {
    if (_current_cacheable) {
        const auto cached = std::find_if(_sequences.begin(), _sequences.end(), [this](const auto& sequence) {
            return sequence_allocations_alive(sequence) && sequence_matches(sequence);
        });
        if (cached != _sequences.end()) {
            ++_hit_count;
            return {cached->command_buffer, true, true, false};
        }
    }

    ++_miss_count;
    _last_miss_submission = submission_sequence;
    const auto cache_capacity = current_cache_capacity();
    if (_current_cacheable && _sequences.size() < cache_capacity) {
        command_sequence sequence;
        sequence.command_buffer = allocate_command_buffer();
        sequence.dispatches = _current_dispatches;
        sequence.barriers = _current_barriers;
        sequence.allocations = _current_allocations;
        record_commands(sequence.command_buffer, sequence.dispatches, sequence.barriers, VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
        _sequences.push_back(std::move(sequence));
        return {_sequences.back().command_buffer, false, true, false};
    }

    record_commands(transient_command_buffer, _current_dispatches, _current_barriers, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    return {transient_command_buffer, false, false, true};
}

void vulkan_command_sequence_cache::discard_expired() {
    for (auto iterator = _sequences.begin(); iterator != _sequences.end();) {
        if (sequence_allocations_alive(*iterator)) {
            ++iterator;
            continue;
        }
        vkFreeCommandBuffers(_device, _command_pool, 1, &iterator->command_buffer);
        iterator = _sequences.erase(iterator);
    }
}

size_t vulkan_command_sequence_cache::current_dispatch_count() const {
    return _current_dispatches.size();
}

size_t vulkan_command_sequence_cache::cached_sequence_count() const {
    return _sequences.size();
}

uint64_t vulkan_command_sequence_cache::hit_count() const {
    return _hit_count;
}

uint64_t vulkan_command_sequence_cache::miss_count() const {
    return _miss_count;
}

uint64_t vulkan_command_sequence_cache::last_miss_submission() const {
    return _last_miss_submission;
}

bool vulkan_command_sequence_cache::barriers_equal(const VkBufferMemoryBarrier2& lhs, const VkBufferMemoryBarrier2& rhs) {
    return lhs.srcStageMask == rhs.srcStageMask && lhs.srcAccessMask == rhs.srcAccessMask && lhs.dstStageMask == rhs.dstStageMask &&
           lhs.dstAccessMask == rhs.dstAccessMask && lhs.srcQueueFamilyIndex == rhs.srcQueueFamilyIndex && lhs.dstQueueFamilyIndex == rhs.dstQueueFamilyIndex &&
           lhs.buffer == rhs.buffer && lhs.offset == rhs.offset && lhs.size == rhs.size;
}

bool vulkan_command_sequence_cache::dispatches_equal(const recorded_dispatch& lhs,
                                                     const std::vector<VkBufferMemoryBarrier2>& lhs_barriers,
                                                     const recorded_dispatch& rhs,
                                                     const std::vector<VkBufferMemoryBarrier2>& rhs_barriers) {
    if (lhs.pipeline->pipeline != rhs.pipeline->pipeline || lhs.pipeline->pipeline_layout != rhs.pipeline->pipeline_layout ||
        lhs.descriptor_set != rhs.descriptor_set || lhs.group_counts != rhs.group_counts || lhs.push_constants != rhs.push_constants ||
        lhs.external_dependency_barrier != rhs.external_dependency_barrier || lhs.barrier_count != rhs.barrier_count) {
        return false;
    }
    for (size_t index = 0; index < lhs.barrier_count; ++index) {
        if (!barriers_equal(lhs_barriers[lhs.barrier_offset + index], rhs_barriers[rhs.barrier_offset + index])) {
            return false;
        }
    }
    return true;
}

bool vulkan_command_sequence_cache::sequence_matches(const command_sequence& sequence) const {
    if (sequence.dispatches.size() != _current_dispatches.size()) {
        return false;
    }
    for (size_t index = 0; index < _current_dispatches.size(); ++index) {
        if (!dispatches_equal(sequence.dispatches[index], sequence.barriers, _current_dispatches[index], _current_barriers)) {
            return false;
        }
    }
    return true;
}

bool vulkan_command_sequence_cache::sequence_allocations_alive(const command_sequence& sequence) {
    return std::all_of(sequence.allocations.begin(), sequence.allocations.end(), [](const auto& allocation) {
        return !allocation.expired();
    });
}

VkCommandBuffer vulkan_command_sequence_cache::allocate_command_buffer() const {
    VkCommandBufferAllocateInfo command_buffer_info{};
    command_buffer_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_info.commandPool = _command_pool;
    command_buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    check_vk_result(vkAllocateCommandBuffers(_device, &command_buffer_info, &command_buffer), "vkAllocateCommandBuffers");
    return command_buffer;
}

void vulkan_command_sequence_cache::record_commands(VkCommandBuffer command_buffer,
                                                    const std::vector<recorded_dispatch>& dispatches,
                                                    const std::vector<VkBufferMemoryBarrier2>& barriers,
                                                    VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = flags;
    check_vk_result(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

    for (const auto& dispatch : dispatches) {
        record_vulkan_synchronization(command_buffer,
                                      dispatch.external_dependency_barrier,
                                      dispatch.barrier_count == 0 ? nullptr : barriers.data() + dispatch.barrier_offset,
                                      dispatch.barrier_count);
        record_vulkan_dispatch(command_buffer, *dispatch.pipeline, dispatch.descriptor_set, dispatch.push_constants, dispatch.group_counts);
    }
    check_vk_result(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
}

size_t vulkan_command_sequence_cache::current_cache_capacity() const {
    size_t descriptor_footprint = 0;
    for (const auto& dispatch : _current_dispatches) {
        const auto dispatch_footprint = std::max<uint32_t>(dispatch.pipeline->descriptor_count, 1);
        if (dispatch_footprint > vulkan_descriptor_cache::capacity || descriptor_footprint > vulkan_descriptor_cache::capacity - dispatch_footprint) {
            return 0;
        }
        descriptor_footprint += dispatch_footprint;
    }
    return vulkan_descriptor_cache::capacity / std::max<size_t>(descriptor_footprint, 1);
}

}  // namespace cldnn::vulkan
