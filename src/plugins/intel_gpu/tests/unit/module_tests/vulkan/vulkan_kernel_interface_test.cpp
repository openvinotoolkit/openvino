// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan/vulkan_kernel_interface.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "foundation_stage_local_spirv.hpp"
#include "foundation_stage_output_spirv.hpp"
#include "openvino/core/except.hpp"
#include "vulkan/vulkan_shader_abi.hpp"

using namespace cldnn::vulkan;

TEST(vulkan_kernel_interface, reflects_foundation_storage_and_local_size_contract) {
    std::vector<uint8_t> spirv(sizeof(foundation_stage_local_spirv));
    std::memcpy(spirv.data(), foundation_stage_local_spirv, spirv.size());

    const auto interface = vulkan_kernel_interface::reflect(spirv, "main");
    ASSERT_EQ(interface.descriptor_bindings.size(), 2);
    EXPECT_EQ(interface.descriptor_bindings[0], (vulkan_descriptor_binding{0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.descriptor_bindings[1], (vulkan_descriptor_binding{0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}));
    EXPECT_EQ(interface.push_constant_size, 0);
    EXPECT_EQ(interface.local_size_defaults, (std::array<uint32_t, 3>{1, 1, 1}));
    ASSERT_TRUE(interface.local_size_specialization_ids[0].has_value());
    EXPECT_EQ(*interface.local_size_specialization_ids[0], shader_abi::index(shader_abi::specialization_id::local_size_x));
    EXPECT_FALSE(interface.local_size_specialization_ids[1].has_value());
    EXPECT_FALSE(interface.local_size_specialization_ids[2].has_value());
    EXPECT_EQ(interface.specialization_ids,
              (std::vector<uint32_t>{shader_abi::index(shader_abi::specialization_id::local_size_x),
                                     shader_abi::index(shader_abi::specialization_id::foundation_local_memory_words)}));
}

TEST(vulkan_kernel_interface, rejects_missing_entry_point) {
    std::vector<uint8_t> spirv(sizeof(foundation_stage_output_spirv));
    std::memcpy(spirv.data(), foundation_stage_output_spirv, spirv.size());
    EXPECT_THROW((void)vulkan_kernel_interface::reflect(spirv, "missing"), ov::Exception);
}
