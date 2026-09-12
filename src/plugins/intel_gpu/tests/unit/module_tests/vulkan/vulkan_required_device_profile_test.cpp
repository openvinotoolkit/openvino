// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan/vulkan_required_device_profile.hpp"

#include <gtest/gtest.h>

using namespace cldnn::vulkan;

namespace {

vulkan_required_device_profile make_supported_profile() {
    vulkan_required_device_profile profile;
    profile.properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    profile.properties.apiVersion = required_api_version;
    profile.storage_buffer_8_bit_access = VK_TRUE;
    profile.synchronization2 = VK_TRUE;
    profile.timeline_semaphore = VK_TRUE;
    profile.maintenance4 = VK_TRUE;
    return profile;
}

}  // namespace

TEST(vulkan_required_device_profile, accepts_complete_profile) {
    EXPECT_TRUE(make_supported_profile().is_supported());
}

TEST(vulkan_required_device_profile, rejects_non_gpu_and_old_api) {
    auto non_gpu = make_supported_profile();
    non_gpu.properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_CPU;
    EXPECT_FALSE(non_gpu.is_supported());

    auto old_api = make_supported_profile();
    old_api.properties.apiVersion = VK_API_VERSION_1_2;
    EXPECT_FALSE(old_api.is_supported());
}

TEST(vulkan_required_device_profile, rejects_each_missing_required_feature) {
    auto profile = make_supported_profile();
    profile.storage_buffer_8_bit_access = VK_FALSE;
    EXPECT_FALSE(profile.is_supported());

    profile = make_supported_profile();
    profile.synchronization2 = VK_FALSE;
    EXPECT_FALSE(profile.is_supported());

    profile = make_supported_profile();
    profile.timeline_semaphore = VK_FALSE;
    EXPECT_FALSE(profile.is_supported());

    profile = make_supported_profile();
    profile.maintenance4 = VK_FALSE;
    EXPECT_FALSE(profile.is_supported());
}
