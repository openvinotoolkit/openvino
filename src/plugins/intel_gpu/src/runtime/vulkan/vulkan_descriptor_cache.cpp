// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_descriptor_cache.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

#include "openvino/core/except.hpp"
#include "vulkan_kernel_arguments.hpp"

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

vulkan_descriptor_cache::vulkan_descriptor_cache(VkDevice device) : _device(device) {
    OPENVINO_ASSERT(_device != VK_NULL_HANDLE, "[GPU][Vulkan] Descriptor cache requires a valid Vulkan device");
}

vulkan_descriptor_cache::~vulkan_descriptor_cache() {
    for (auto& entry : _pools) {
        for (auto& block : entry.second) {
            if (block.pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(_device, block.pool, nullptr);
            }
        }
    }
}

bool vulkan_descriptor_cache::descriptor_key::operator<(const descriptor_key& other) const {
    if (layout != other.layout) {
        return std::less<VkDescriptorSetLayout>{}(layout, other.layout);
    }
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

bool vulkan_descriptor_cache::requires_mutable_set(VkDescriptorSetLayout layout, const vulkan_prepared_arguments& prepared) const {
    return _sets.size() >= capacity && _sets.find(make_key(layout, prepared)) == _sets.end();
}

vulkan_descriptor_cache::lookup_result vulkan_descriptor_cache::lookup_or_allocate(VkDescriptorSetLayout layout, const vulkan_prepared_arguments& prepared) {
    auto key = make_key(layout, prepared);
    const auto cached = _sets.find(key);
    if (cached != _sets.end()) {
        return {cached->second, lookup_status::reused};
    }
    if (_sets.size() >= capacity) {
        return {};
    }

    const auto descriptor_count = checked_u32(prepared.buffer_infos.size(), "descriptor count");
    auto* const descriptor_set = allocate(layout, descriptor_count);
    update(descriptor_set, prepared.buffer_infos);
    _sets.emplace(std::move(key), descriptor_set);
    return {descriptor_set, lookup_status::allocated};
}

void vulkan_descriptor_cache::update(VkDescriptorSet descriptor_set, const std::vector<VkDescriptorBufferInfo>& buffer_infos) const {
    std::vector<VkWriteDescriptorSet> descriptor_writes(buffer_infos.size());
    for (uint32_t index = 0; index < descriptor_writes.size(); ++index) {
        descriptor_writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[index].dstSet = descriptor_set;
        descriptor_writes[index].dstBinding = index;
        descriptor_writes[index].descriptorCount = 1;
        descriptor_writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[index].pBufferInfo = &buffer_infos[index];
    }
    vkUpdateDescriptorSets(_device, static_cast<uint32_t>(descriptor_writes.size()), descriptor_writes.data(), 0, nullptr);
}

vulkan_descriptor_cache::descriptor_key vulkan_descriptor_cache::make_key(VkDescriptorSetLayout layout, const vulkan_prepared_arguments& prepared) {
    descriptor_key key;
    key.layout = layout;
    key.buffer_infos = prepared.buffer_infos;
    key.allocations.reserve(prepared.allocations.size());
    for (const auto& allocation : prepared.allocations) {
        key.allocations.emplace_back(allocation);
    }
    return key;
}

VkDescriptorSet vulkan_descriptor_cache::allocate(VkDescriptorSetLayout layout, uint32_t descriptor_count) {
    auto& blocks = _pools[descriptor_count];
    if (blocks.empty() || blocks.back().allocated_sets == blocks.back().capacity) {
        const auto remaining_cache_entries = capacity - _sets.size();
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
        check_vk_result(vkCreateDescriptorPool(_device, &pool_info, nullptr, &block.pool), "vkCreateDescriptorPool(cache)");
        blocks.push_back(block);
    }

    auto& block = blocks.back();
    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = block.pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &layout;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    check_vk_result(vkAllocateDescriptorSets(_device, &allocate_info, &descriptor_set), "vkAllocateDescriptorSets(cache)");
    ++block.allocated_sets;
    return descriptor_set;
}

}  // namespace cldnn::vulkan
