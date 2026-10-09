// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#ifdef OV_GPU_WITH_SYCL_RT

#include "sycl_test_context.hpp"

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace cldnn;
using namespace sycl_tests;

// Build a trivial OpenCL C kernel from source via the SYCL kernel builder.
TEST(sycl_kernel_builder, build_from_source) {
    auto ctx = create_sycl_test_context();
    auto builder = ctx.sycl_test_engine->create_kernel_builder();
    ASSERT_NE(builder, nullptr);

    const std::string source = R"__cl(
        __kernel void test_add(__global const float* a,
                               __global const float* b,
                               __global float* c) {
            const uint i = get_global_id(0);
            c[i] = a[i] + b[i];
        }
    )__cl";

    std::vector<kernel::ptr> kernels;
    ASSERT_NO_THROW(builder->build_kernels(source.data(), source.size(),
                                           KernelFormat::SOURCE, "",
                                           kernels));
    ASSERT_EQ(kernels.size(), 1);
    EXPECT_EQ(kernels[0]->get_id(), "test_add");
}

// Verify the SOURCE -> get_binary() -> NATIVE_BIN roundtrip:
// a kernel compiled from source should produce a binary (SPIR-V) that can
// be fed back through build_kernels with NATIVE_BIN to recreate an
// equivalent kernel.
TEST(sycl_kernel_builder, build_from_cached_binary) {
    auto ctx = create_sycl_test_context();
    auto builder = ctx.sycl_test_engine->create_kernel_builder();
    ASSERT_NE(builder, nullptr);

    const std::string source = R"__cl(
        __kernel void test_mul(__global const float* in,
                               __global float* out) {
            const uint i = get_global_id(0);
            out[i] = in[i] * 2.0f;
        }
    )__cl";

    // Step 1: Build from source
    std::vector<kernel::ptr> source_kernels;
    ASSERT_NO_THROW(builder->build_kernels(source.data(), source.size(),
                                           KernelFormat::SOURCE, "",
                                           source_kernels));
    ASSERT_EQ(source_kernels.size(), 1);
    EXPECT_EQ(source_kernels[0]->get_id(), "test_mul");

    // Step 2: Extract binary (SPIR-V)
    std::vector<uint8_t> binary = source_kernels[0]->get_binary();
    ASSERT_FALSE(binary.empty());

    // Step 3: Rebuild from NATIVE_BIN
    std::vector<kernel::ptr> cached_kernels;
    ASSERT_NO_THROW(builder->build_kernels(binary.data(), binary.size(),
                                           KernelFormat::NATIVE_BIN, "",
                                           cached_kernels));
    ASSERT_EQ(cached_kernels.size(), 1);
    EXPECT_EQ(cached_kernels[0]->get_id(), source_kernels[0]->get_id());

    // Step 4: The rebuilt kernel should also produce a valid binary
    std::vector<uint8_t> binary2 = cached_kernels[0]->get_binary();
    EXPECT_EQ(binary, binary2);
}

#endif  // OV_GPU_WITH_SYCL_RT
