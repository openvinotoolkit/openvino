// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan/vulkan_kernel_interface.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"
#include "slang_bootstrap_spirv.hpp"

using namespace cldnn::vulkan;

TEST(vulkan_kernel_interface, reflects_slang_storage_push_constant_and_fixed_local_size_contract) {
    std::vector<uint8_t> spirv(sizeof(slang_bootstrap_spirv));
    std::memcpy(spirv.data(), slang_bootstrap_spirv, spirv.size());

    const auto interface = vulkan_kernel_interface::reflect(spirv, "slang_bootstrap");
    ASSERT_EQ(interface.descriptor_bindings.size(), 2);
    EXPECT_EQ(interface.descriptor_bindings[0], (vulkan_descriptor_binding{0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.descriptor_bindings[1], (vulkan_descriptor_binding{0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.push_constant_size, sizeof(float));
    EXPECT_EQ(interface.local_size_defaults, (std::array<uint32_t, 3>{64, 1, 1}));
    EXPECT_FALSE(interface.local_size_specialization_ids[0].has_value());
    EXPECT_FALSE(interface.local_size_specialization_ids[1].has_value());
    EXPECT_FALSE(interface.local_size_specialization_ids[2].has_value());
    EXPECT_TRUE(interface.specialization_ids.empty());
}

TEST(vulkan_kernel_interface, rejects_missing_entry_point) {
    std::vector<uint8_t> spirv(sizeof(slang_bootstrap_spirv));
    std::memcpy(spirv.data(), slang_bootstrap_spirv, spirv.size());
    EXPECT_THROW((void)vulkan_kernel_interface::reflect(spirv, "missing"), ov::Exception);
}
