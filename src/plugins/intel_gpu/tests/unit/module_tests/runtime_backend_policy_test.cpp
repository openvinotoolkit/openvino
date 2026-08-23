// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

using namespace cldnn;

TEST(runtime_backend_policy, defaults_to_native_cached_kernel_artifacts) {
    const gpu_kernel_cache_policy policy;

    EXPECT_EQ(policy.artifact, gpu_cached_kernel_artifact::native_device_binary);
}

TEST(runtime_backend_policy, compiled_backends_declare_cached_kernel_artifact) {
    for (const auto& backend : runtime_backend_registry::compiled_backends()) {
        const auto expected =
            backend.runtime_type == runtime_types::vulkan ? gpu_cached_kernel_artifact::spirv : gpu_cached_kernel_artifact::native_device_binary;
        EXPECT_EQ(backend.kernel_cache.artifact, expected) << backend.name;
    }
}
