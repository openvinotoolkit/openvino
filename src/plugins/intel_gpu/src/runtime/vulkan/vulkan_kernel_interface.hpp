// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cldnn::vulkan {

struct vulkan_descriptor_binding final {
    uint32_t set = 0;
    uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;

    bool operator==(const vulkan_descriptor_binding& other) const;
    bool operator<(const vulkan_descriptor_binding& other) const;
};

/// Vulkan-local, source-compiler-independent compute interface reflected from SPIR-V.
struct vulkan_kernel_interface final {
    std::vector<vulkan_descriptor_binding> descriptor_bindings;
    uint32_t push_constant_size = 0;
    std::array<uint32_t, 3> local_size_defaults{1, 1, 1};
    std::array<std::optional<uint32_t>, 3> local_size_specialization_ids;
    std::vector<uint32_t> specialization_ids;

    static std::string get_single_entry_point(const std::vector<uint8_t>& spirv);
    static vulkan_kernel_interface reflect(const std::vector<uint8_t>& spirv, const std::string& entry_point);
    void validate_canonical_compute_abi(const std::string& entry_point) const;
};

}  // namespace cldnn::vulkan
