// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "vulkan_memory.hpp"

namespace cldnn::vulkan {

struct vulkan_prepared_arguments;

class vulkan_descriptor_cache final {
public:
    static constexpr size_t capacity = 256;

    enum class lookup_status {
        reused,
        allocated,
        unavailable,
    };

    struct lookup_result final {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        lookup_status status = lookup_status::unavailable;
    };

    explicit vulkan_descriptor_cache(VkDevice device);
    ~vulkan_descriptor_cache();

    vulkan_descriptor_cache(const vulkan_descriptor_cache&) = delete;
    vulkan_descriptor_cache& operator=(const vulkan_descriptor_cache&) = delete;

    bool requires_mutable_set(const vulkan_prepared_arguments& prepared) const;
    lookup_result lookup_or_allocate(VkDescriptorSetLayout layout, const vulkan_prepared_arguments& prepared);
    void update(VkDescriptorSet descriptor_set, const std::vector<VkDescriptorBufferInfo>& buffer_infos) const;

private:
    struct descriptor_key final {
        std::vector<std::weak_ptr<vulkan_buffer_allocation>> allocations;
        std::vector<VkDescriptorBufferInfo> buffer_infos;

        bool operator<(const descriptor_key& other) const;
    };

    struct descriptor_pool_block final {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        uint32_t capacity = 0;
        uint32_t allocated_sets = 0;
    };

    static descriptor_key make_key(const vulkan_prepared_arguments& prepared);
    VkDescriptorSet allocate(VkDescriptorSetLayout layout, uint32_t descriptor_count);

    VkDevice _device = VK_NULL_HANDLE;
    std::map<descriptor_key, VkDescriptorSet> _sets;
    std::map<uint32_t, std::vector<descriptor_pool_block>> _pools;
};

}  // namespace cldnn::vulkan
