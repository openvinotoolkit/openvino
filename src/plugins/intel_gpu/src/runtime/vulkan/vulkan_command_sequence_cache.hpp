// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "vulkan_memory.hpp"

namespace cldnn::vulkan {

struct vulkan_pipeline_state;

void record_vulkan_synchronization(VkCommandBuffer command_buffer,
                                   bool external_dependency_barrier,
                                   const VkBufferMemoryBarrier2* barriers,
                                   size_t barrier_count);

void record_vulkan_dispatch(VkCommandBuffer command_buffer,
                            const vulkan_pipeline_state& pipeline,
                            VkDescriptorSet descriptor_set,
                            const std::vector<uint8_t>& push_constants,
                            const std::array<uint32_t, 3>& group_counts);

class vulkan_command_sequence_cache final {
public:
    struct resolution final {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        bool cache_hit = false;
        bool replayable = false;
        bool uses_transient_buffer = false;
    };

    vulkan_command_sequence_cache(VkDevice device, uint32_t queue_family_index);
    ~vulkan_command_sequence_cache();

    vulkan_command_sequence_cache(const vulkan_command_sequence_cache&) = delete;
    vulkan_command_sequence_cache& operator=(const vulkan_command_sequence_cache&) = delete;

    void begin_batch();
    void append(std::shared_ptr<const vulkan_pipeline_state> pipeline,
                VkDescriptorSet descriptor_set,
                std::vector<uint8_t>&& push_constants,
                const std::array<uint32_t, 3>& group_counts,
                const std::vector<VkBufferMemoryBarrier2>& barriers,
                bool external_dependency_barrier,
                const std::vector<vulkan_buffer_allocation::ptr>& allocations,
                bool descriptor_is_immutable);
    resolution resolve(VkCommandBuffer transient_command_buffer, uint64_t submission_sequence);
    void discard_expired();

    size_t current_dispatch_count() const;
    size_t cached_sequence_count() const;
    uint64_t hit_count() const;
    uint64_t miss_count() const;
    uint64_t last_miss_submission() const;

private:
    struct recorded_dispatch final {
        std::shared_ptr<const vulkan_pipeline_state> pipeline;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        std::array<uint32_t, 3> group_counts{};
        std::vector<uint8_t> push_constants;
        size_t barrier_offset = 0;
        size_t barrier_count = 0;
        bool external_dependency_barrier = false;
    };

    struct command_sequence final {
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        std::vector<recorded_dispatch> dispatches;
        std::vector<VkBufferMemoryBarrier2> barriers;
        std::vector<std::weak_ptr<vulkan_buffer_allocation>> allocations;
    };

    static bool barriers_equal(const VkBufferMemoryBarrier2& lhs, const VkBufferMemoryBarrier2& rhs);
    static bool dispatches_equal(const recorded_dispatch& lhs,
                                 const std::vector<VkBufferMemoryBarrier2>& lhs_barriers,
                                 const recorded_dispatch& rhs,
                                 const std::vector<VkBufferMemoryBarrier2>& rhs_barriers);
    bool sequence_matches(const command_sequence& sequence) const;
    static bool sequence_allocations_alive(const command_sequence& sequence);

    VkCommandBuffer allocate_command_buffer() const;
    static void record_commands(VkCommandBuffer command_buffer,
                                const std::vector<recorded_dispatch>& dispatches,
                                const std::vector<VkBufferMemoryBarrier2>& barriers,
                                VkCommandBufferUsageFlags flags);
    size_t current_cache_capacity() const;

    VkDevice _device = VK_NULL_HANDLE;
    VkCommandPool _command_pool = VK_NULL_HANDLE;
    std::vector<command_sequence> _sequences;
    std::vector<recorded_dispatch> _current_dispatches;
    std::vector<VkBufferMemoryBarrier2> _current_barriers;
    std::vector<std::weak_ptr<vulkan_buffer_allocation>> _current_allocations;
    uint64_t _hit_count = 0;
    uint64_t _miss_count = 0;
    uint64_t _last_miss_submission = 0;
    bool _current_cacheable = true;
};

}  // namespace cldnn::vulkan
