// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "vulkan_memory.hpp"

namespace cldnn::vulkan {

struct vulkan_prepared_arguments;

class vulkan_buffer_hazard_tracker final {
public:
    void record(const vulkan_prepared_arguments& prepared, VkPipelineStageFlags2 stages);
    void mark_external_dependency();
    void clear();

    const std::vector<VkBufferMemoryBarrier2>& barriers() const;
    bool has_external_dependency_barrier() const;

    uint64_t argument_access_count() const;
    uint64_t barrier_count() const;
    uint64_t external_dependency_barrier_count() const;

private:
    struct access_range final {
        VkDeviceSize begin = 0;
        VkDeviceSize end = 0;
        VkPipelineStageFlags2 stages = 0;
        VkAccessFlags2 access = 0;
        uint64_t generation = 0;
    };

    struct allocation_access_state final {
        std::vector<access_range> ranges;
    };

    using allocation_key = std::weak_ptr<vulkan_buffer_allocation>;
    using allocation_access_map = std::map<allocation_key, allocation_access_state, std::owner_less<allocation_key>>;

    static const access_range* range_for_interval(const std::vector<access_range>& ranges, VkDeviceSize begin, VkDeviceSize end);
    static access_range make_access_range(const VkDescriptorBufferInfo& info, VkPipelineStageFlags2 stages, VkAccessFlags2 access, uint64_t generation);
    static void update_access_state(std::vector<access_range>& ranges, const access_range& current);
    static void append_barriers(const vulkan_buffer_allocation::ptr& allocation,
                                const std::vector<access_range>& previous,
                                const access_range& current,
                                std::vector<VkBufferMemoryBarrier2>& barriers);

    void clear_access_history();
    void advance_generation();
    void discard_expired_accesses();

    allocation_access_map _access_states;
    std::vector<VkBufferMemoryBarrier2> _barriers;
    std::vector<access_range> _current_accesses;
    std::vector<allocation_access_state*> _access_state_scratch;
    uint64_t _generation = 0;
    uint64_t _argument_access_count = 0;
    uint64_t _barrier_count = 0;
    uint64_t _external_dependency_barrier_count = 0;
    bool _external_dependency_pending = false;
    bool _has_external_dependency_barrier = false;
};

}  // namespace cldnn::vulkan
