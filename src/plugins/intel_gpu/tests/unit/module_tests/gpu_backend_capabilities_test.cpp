// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/gpu_backend_capabilities.hpp"

#include <gtest/gtest.h>

using namespace cldnn;

namespace {
constexpr uint32_t tested_tensor_rank_limit = 8;
}  // namespace

TEST(gpu_backend_capabilities, separates_storage_native_arithmetic_and_emulation) {
    gpu_backend_capabilities capabilities;
    capabilities.legacy_device_info_adapter = false;
    capabilities.fp16 = {true, gpu_arithmetic_support::emulated};
    capabilities.fp32 = {true, gpu_arithmetic_support::native};

    EXPECT_TRUE(capabilities.fp16.storage);
    EXPECT_EQ(capabilities.fp16.arithmetic, gpu_arithmetic_support::emulated);
    EXPECT_NE(capabilities.fp16.arithmetic, gpu_arithmetic_support::native);
    EXPECT_TRUE(capabilities.fp32.storage);
    EXPECT_EQ(capabilities.fp32.arithmetic, gpu_arithmetic_support::native);
}

TEST(gpu_backend_capabilities, exposes_backend_neutral_layout_classes) {
    gpu_layout_capabilities layouts;
    layouts.dense_buffers = true;
    layouts.strided_buffers = true;
    layouts.max_tensor_rank = tested_tensor_rank_limit;

    EXPECT_TRUE(layouts.supports(gpu_layout_kind::dense_buffer));
    EXPECT_TRUE(layouts.supports(gpu_layout_kind::strided_buffer));
    EXPECT_FALSE(layouts.supports(gpu_layout_kind::blocked_buffer));
    EXPECT_FALSE(layouts.supports(gpu_layout_kind::image));
    EXPECT_FALSE(layouts.supports(gpu_layout_kind::planar_image));
    EXPECT_EQ(layouts.max_tensor_rank, tested_tensor_rank_limit);
}

TEST(gpu_backend_capabilities, prevents_portable_backends_from_inheriting_blocked_layout_policy) {
    gpu_backend_capabilities capabilities;
    capabilities.legacy_device_info_adapter = false;
    capabilities.layouts.dense_buffers = true;
    capabilities.layouts.strided_buffers = true;

    EXPECT_TRUE(capabilities.layouts.supports(gpu_layout_kind::dense_buffer));
    EXPECT_TRUE(capabilities.layouts.supports(gpu_layout_kind::strided_buffer));
    EXPECT_FALSE(capabilities.layouts.supports(gpu_layout_kind::blocked_buffer));
}

TEST(gpu_backend_capabilities, keeps_legacy_backends_on_compatibility_adapter) {
    const gpu_backend_capabilities capabilities;

    EXPECT_TRUE(capabilities.legacy_device_info_adapter);
    EXPECT_EQ(capabilities.execution_tier, gpu_execution_tier::legacy);
    EXPECT_EQ(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::native_device_binary);
}

TEST(gpu_backend_capabilities, distinguishes_portable_cached_kernel_artifacts) {
    gpu_backend_capabilities capabilities;
    capabilities.kernel_cache.artifact = gpu_cached_kernel_artifact::spirv;

    EXPECT_EQ(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::spirv);
    EXPECT_NE(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::native_device_binary);
}
