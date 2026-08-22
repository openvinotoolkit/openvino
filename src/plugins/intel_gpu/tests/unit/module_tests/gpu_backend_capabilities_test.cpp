// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/gpu_backend_capabilities.hpp"

#include <gtest/gtest.h>

using namespace cldnn;

TEST(gpu_backend_capabilities, defaults_to_native_cached_kernel_artifacts) {
    const gpu_backend_capabilities capabilities;

    EXPECT_EQ(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::native_device_binary);
}

TEST(gpu_backend_capabilities, distinguishes_portable_cached_kernel_artifacts) {
    gpu_backend_capabilities capabilities;
    capabilities.kernel_cache.artifact = gpu_cached_kernel_artifact::spirv;

    EXPECT_EQ(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::spirv);
    EXPECT_NE(capabilities.kernel_cache.artifact, gpu_cached_kernel_artifact::native_device_binary);
}
