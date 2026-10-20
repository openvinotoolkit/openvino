// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"

using cldnn::MOECompressed;

TEST(moe_offload_primitive_test, default_offload_fields) {
    cldnn::moe_3gemm_fused_compressed prim;
    ASSERT_TRUE(prim._otd.weight_bin_offsets.empty());
    ASSERT_TRUE(prim._otd.weights_path.empty());
    ASSERT_EQ(prim._otd.lru_expert_num, 0U);
}

TEST(moe_offload_primitive_test, construct_with_offload_params) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    std::string path = "/path/to/weights.bin";
    size_t lru_num = 16;

    cldnn::moe_3gemm_fused_compressed prim(
        "test_moe",
        {cldnn::input_info("input0"), cldnn::input_info("input1")},
        config,
        offsets,
        path,
        lru_num);

    ASSERT_EQ(prim._otd.weight_bin_offsets.size(), 9U);
    ASSERT_EQ(prim._otd.weight_bin_offsets[0], 0U);
    ASSERT_EQ(prim._otd.weight_bin_offsets[8], 800U);
    ASSERT_EQ(prim._otd.weights_path, path);
    ASSERT_EQ(prim._otd.lru_expert_num, lru_num);
}

TEST(moe_offload_primitive_test, equality_with_offload_fields) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};

    cldnn::moe_3gemm_fused_compressed prim1(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        16);

    cldnn::moe_3gemm_fused_compressed prim2(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        16);

    ASSERT_TRUE(prim1 == prim2);

    // Different weights_path -> not equal
    cldnn::moe_3gemm_fused_compressed prim3(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/b.bin",
        16);
    ASSERT_FALSE(prim1 == prim3);

    // Different lru_expert_num -> not equal
    cldnn::moe_3gemm_fused_compressed prim4(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        32);
    ASSERT_FALSE(prim1 == prim4);

    // Different offsets -> not equal
    std::vector<size_t> offsets2 = {0, 100, 200, 300, 400, 500, 600, 700, 999};
    cldnn::moe_3gemm_fused_compressed prim5(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets2,
        "/path/a.bin",
        16);
    ASSERT_FALSE(prim1 == prim5);
}
