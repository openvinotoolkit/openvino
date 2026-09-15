// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/runtime/runtime_backend_registry.hpp"
#include "registry/runtime_implementation_policy.hpp"

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

TEST(runtime_implementation_policy, rejects_cross_runtime_device_implementations) {
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::ocl, false));
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::sycl, false));
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::ocl, impl_types::vulkan, false));
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::ze, impl_types::vulkan, false));
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::sycl, impl_types::vulkan, false));
}

TEST(runtime_implementation_policy, permits_only_shape_flow_host_implementation_for_vulkan) {
    EXPECT_FALSE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::cpu, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::cpu, true));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::common, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::vulkan, impl_types::vulkan, false));
}

TEST(runtime_implementation_policy, preserves_established_non_vulkan_implementations) {
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::ocl, impl_types::ocl, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::ze, impl_types::ocl, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::sycl, impl_types::sycl, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::ocl, impl_types::onednn, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::ocl, impl_types::cm, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::sycl, impl_types::onednn, false));
    EXPECT_TRUE(runtime_implementation_policy::allows(runtime_types::ocl, impl_types::cpu, false));
}
