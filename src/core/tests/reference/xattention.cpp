// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iterator>
#include <openvino/reference/xattention.hpp>

double DEFAULT_THRESHOLD = 0.8;
size_t DEFAULT_BLOCK_SIZE = 32;
size_t DEFAULT_STRIDE = 8;

TEST(XAttentionBasicTest, SelectsBlocksWithoutThrowing) {
    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD, DEFAULT_BLOCK_SIZE, DEFAULT_STRIDE);

    ov::Shape q_shape = {2, 64, 32};
    ov::Shape k_shape = {2, 128, 32};
    std::vector<double> q_data(ov::shape_size(q_shape), 1.0);
    std::vector<double> k_data(ov::shape_size(k_shape), 1.0);
    EXPECT_NO_THROW(selector.select_blocks(q_data.data(), q_shape, k_data.data(), k_shape));
};

struct DiagonalReshapeTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    bool is_antidiagonal;
    size_t block_size;
    size_t stride;
    ov::Shape out_shape;
    std::vector<double> ref_out_data;
};

using XAttentionDiagonalReshapeTest = ::testing::TestWithParam<DiagonalReshapeTestData>;

std::vector<DiagonalReshapeTestData> DIAGONAL_RESHAPE_TEST_CASES = {
    {
        {2, 4, 4},
        // clang-format off
        {
             3.144,  8.512,  8.518, -8.386,
             7.889, -5.721,  5.507,  4.295,
            -6.624, -8.463,  7.474,  9.879,
             4.534, -5.908, -9.388,  2.356,

             7.497,  8.186, -8.658, -4.796,
            -8.248, -9.797, -7.907, -4.513,
             3.469,  7.633,  7.244, -6.844,
            -7.173,  4.450,  6.705, -7.035
        },
        // clang-format on

        /* is_antidiagonal = */ true,
        /* block_size = */ 2,
        /* stride = */ 2,
        {2, 2, 8},

        // clang-format off
        {
             7.889, -5.721,  5.507,  4.295,  3.144,  8.512,  8.518, -8.386,
             4.534, -5.908, -9.388,  2.356, -6.624, -8.463,  7.474,  9.879,

            -8.248, -9.797, -7.907, -4.513,  7.497,  8.186, -8.658, -4.796,
            -7.173,  4.450,  6.705, -7.035,  3.469,  7.633,  7.244, -6.844,
        },
        // clang-format on
    },
    {
        {2, 4, 4},
        // clang-format off
        {
             3.144,  8.512,  8.518, -8.386,
             7.889, -5.721,  5.507,  4.295,
            -6.624, -8.463,  7.474,  9.879,
             4.534, -5.908, -9.388,  2.356,

             7.497,  8.186, -8.658, -4.796,
            -8.248, -9.797, -7.907, -4.513,
             3.469,  7.633,  7.244, -6.844,
            -7.173,  4.450,  6.705, -7.035
        },
        // clang-format on

        /* is_antidiagonal = */ false,
        /* block_size = */ 2,
        /* stride = */ 2,
        {2, 2, 8},

        // clang-format off
        {
             3.144,  8.512,  8.518, -8.386,  7.889, -5.721,  5.507,  4.295,
            -6.624, -8.463,  7.474,  9.879,  4.534, -5.908, -9.388,  2.356,

             7.497,  8.186, -8.658, -4.796, -8.248, -9.797, -7.907, -4.513,
             3.469,  7.633,  7.244, -6.844, -7.173,  4.450,  6.705, -7.035
        },
        // clang-format on
    },
    {
        {2, 9, 2},
        // clang-format off
        {
             1.110, -4.244,
             3.530, -1.083,
             3.664, -2.459,
             3.930, -2.122,
            -4.142,  2.837,
            -7.413,  5.855,
             1.354, -7.748,
             0.264,  7.095,
            -8.410,  6.247,

            -7.832,  9.163,
            -7.414, -3.682,
            -5.429,  7.854,
             1.767,  5.950,
            -0.841,  1.935,
             3.568,  8.530,
             9.438, -2.421,
            -5.892,  7.820,
            -9.869, -7.636
        },
        // clang-format on

        /* is_antidiagonal = */ true,
        /* block_size = */ 9,
        /* stride = */ 3,
        {2, 3, 6},

        // clang-format off
        {
             3.664, -2.459,  3.530, -1.083, 1.110, -4.244,
            -7.413,  5.855, -4.142,  2.837, 3.930, -2.122,
            -8.410,  6.247,  0.264,  7.095, 1.354, -7.748,

            -5.429,  7.854, -7.414, -3.682, -7.832,  9.163,
             3.568,  8.530, -0.841,  1.935, 1.767,  5.950,
            -9.869, -7.636, -5.892,  7.820, 9.438, -2.421,
        },
        // clang-format on
    },
    {
        {2, 9, 2},
        // clang-format off
        {
             1.110, -4.244,
             3.530, -1.083,
             3.664, -2.459,
             3.930, -2.122,
            -4.142,  2.837,
            -7.413,  5.855,
             1.354, -7.748,
             0.264,  7.095,
            -8.410,  6.247,

            -7.832,  9.163,
            -7.414, -3.682,
            -5.429,  7.854,
             1.767,  5.950,
            -0.841,  1.935,
             3.568,  8.530,
             9.438, -2.421,
            -5.892,  7.820,
            -9.869, -7.636
        },
        // clang-format on

        /* is_antidiagonal = */ false,
        /* block_size = */ 9,
        /* stride = */ 3,
        {2, 3, 6},

        // clang-format off
        {
             1.110, -4.244,  3.530, -1.083,  3.664, -2.459,
             3.930, -2.122, -4.142,  2.837, -7.413,  5.855,
             1.354, -7.748,  0.264,  7.095, -8.410,  6.247,

            -7.832,  9.163, -7.414, -3.682, -5.429,  7.854,
             1.767,  5.950, -0.841,  1.935,  3.568,  8.530,
             9.438, -2.421, -5.892,  7.820, -9.869, -7.636
        },
        // clang-format on
    },
};

TEST_P(XAttentionDiagonalReshapeTest, ReshapesDiagonally) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.out_shape));

    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD,
                                                            test_struct.block_size,
                                                            test_struct.stride);
    std::vector<double> test_out_data(test_struct.ref_out_data.size());
    selector.diagonal_reshape(test_struct.in_data.data(),
                              test_struct.in_shape,
                              test_out_data.data(),
                              test_struct.out_shape,
                              test_struct.is_antidiagonal);
    EXPECT_EQ(test_out_data, test_struct.ref_out_data);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         XAttentionDiagonalReshapeTest,
                         ::testing::ValuesIn(DIAGONAL_RESHAPE_TEST_CASES));

struct TransposeMatmulScaleTestData {
    ov::Shape reshaped_query_shape;
    std::vector<double> reshaped_query_data;
    ov::Shape reshaped_key_shape;
    std::vector<double> reshaped_key_data;
    size_t block_size;
    size_t stride;
    ov::Shape out_shape;
    std::vector<double> ref_out_data;
};

using XAttentionTransposeMatmulScaleTest = ::testing::TestWithParam<TransposeMatmulScaleTestData>;

std::vector<TransposeMatmulScaleTestData> TRANSPOSE_MATMUL_SCALE_TEST_CASES = {
    {
        {2, 2, 8},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356, -6.624, -8.463,  7.474,  9.879,
             7.889, -5.721,  5.507,  4.295,  3.144,  8.512,  8.518, -8.386,

            -7.173,  4.450,  6.705, -7.035,  3.469,  7.633,  7.244, -6.844,
            -8.248, -9.797, -7.907, -4.513,  7.497,  8.186, -8.658, -4.796,
        },
        // clang-format on

        {2, 3, 8},

        // clang-format off
        {
            -2.731, -0.545,  6.128, -6.175, -2.198, -1.275, -8.617, -0.683,
             3.085,  7.929, -1.127,  5.369, -6.891,  9.582, -6.954,  1.189,
            -0.610, -6.310, -9.216, -1.196,  9.509, -8.119,  4.652, -4.435,

            -0.026, -9.294,  7.862,  9.318, -6.012,  8.252, -3.224, -0.710,
            -2.915, -7.362, -5.553,  0.097, -4.509,  6.993,  2.021,  2.870,
            -3.682,  8.637, -9.922, -6.336, -2.949,  4.339, -2.807, -9.192
        },

        /* block_size = */ 2,
        /* stride = */ 2,
        {2, 2, 3},

        // clang-format off
        {
            -31.760349,   -21.32551225, 28.723734,
            -24.15923075, -3.369805999, 3.2507255,

            -7.593187497, -4.258293245, 27.08950801,
             10.21206450, 32.95415775, 33.649577
        },
        // clang-format on
    },
};

TEST_P(XAttentionTransposeMatmulScaleTest, TransposesMatmulsAndScales) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.reshaped_key_data.size(), ov::shape_size(test_struct.reshaped_key_shape));
    ASSERT_EQ(test_struct.reshaped_query_data.size(), ov::shape_size(test_struct.reshaped_query_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.out_shape));

    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD,
                                                            test_struct.block_size,
                                                            test_struct.stride);
    std::vector<double> test_out_data(test_struct.ref_out_data.size());
    selector.transpose_matmul_scale(test_struct.reshaped_query_data.data(),
                                    test_struct.reshaped_key_data.data(),
                                    test_struct.reshaped_query_shape,
                                    test_struct.reshaped_key_shape,
                                    test_out_data.data(),
                                    test_struct.out_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-8), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         XAttentionTransposeMatmulScaleTest,
                         ::testing::ValuesIn(TRANSPOSE_MATMUL_SCALE_TEST_CASES));

struct SoftmaxTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    ov::Shape out_shape;
    std::vector<double> ref_out_data;
};

using XAttentionSoftmaxTest = ::testing::TestWithParam<SoftmaxTestData>;

std::vector<SoftmaxTestData> SOFTMAX_TEST_CASES = {
    {
        {2, 2, 4},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356,
             7.889, -5.721,  5.507,  4.295,

            -7.173,  4.450,  6.705, -7.035,
            -8.248, -9.797, -7.907, -4.513
        },
        // clang-format on

        {2, 2, 4},

        // clang-format off
        {
            0.898232, 2.62111e-05, 8.07497e-07, 0.101741,
            0.892973, 1.09671e-06, 0.08248, 0.0245462,

            8.50252e-07, 0.0949189, 0.905079, 9.76069e-07,
            0.0224685, 0.00477366, 0.0315986, 0.941159
        },
    },
};

TEST_P(XAttentionSoftmaxTest, SoftmaxIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.out_shape));

    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD,
                                                            DEFAULT_BLOCK_SIZE,
                                                            DEFAULT_STRIDE);
    std::vector<double> test_out_data(test_struct.ref_out_data.size());
    selector.softmax(test_struct.in_data.data(), test_struct.in_shape, test_out_data.data(), test_struct.out_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-5), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         XAttentionSoftmaxTest,
                         ::testing::ValuesIn(SOFTMAX_TEST_CASES));

struct CausalMaskTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    std::vector<double> ref_out_data;
};

using XAttentionCausalMaskTest = ::testing::TestWithParam<CausalMaskTestData>;

std::vector<CausalMaskTestData> CAUSAL_MASK_TEST_CASES = {
    {
        {2, 4, 4},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356,
             7.889, -5.721,  5.507,  4.295,
             4.534, -5.908, -9.388,  2.356,
             7.889, -5.721,  5.507,  4.295,

            -7.173,  4.450,  6.705, -7.035,
            -8.248, -9.797, -7.907, -4.513,
            -7.173,  4.450,  6.705, -7.035,
            -8.248, -9.797, -7.907, -4.513
        },
        // clang-format on

        // clang-format off
        {
             4.534, -INFINITY, -INFINITY, -INFINITY,
             7.889, -5.721, -INFINITY, -INFINITY,
             4.534, -5.908, -9.388,  -INFINITY,
             7.889, -5.721,  5.507,  4.295,

            -7.173, -INFINITY, -INFINITY, -INFINITY,
            -8.248, -9.797, -INFINITY, -INFINITY,
            -7.173,  4.450,  6.705, -INFINITY,
            -8.248, -9.797, -7.907, -4.513
        },
    },
    {
        {2, 2, 4},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356,
             7.889, -5.721,  5.507,  4.295,

            -7.173,  4.450,  6.705, -7.035,
            -8.248, -9.797, -7.907, -4.513
        },
        // clang-format on

        // clang-format off
        {
             4.534, -5.908, -9.388,  -INFINITY,
             7.889, -5.721,  5.507,  4.295,

            -7.173,  4.450,  6.705, -INFINITY,
            -8.248, -9.797, -7.907, -4.513
        },
    },
    {
        {2, 4, 6},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356, -5.908, -9.388,
             7.889, -5.721,  5.507,  4.295, -5.721,  5.507,
             4.534, -5.908, -9.388,  2.356, -5.908, -9.388,
             7.889, -5.721,  5.507,  4.295, -5.721,  5.507,

            -7.173,  4.450,  6.705, -7.035,  4.450,  6.705,
            -8.248, -9.797, -7.907, -4.513, -9.797, -7.907,
            -7.173,  4.450,  6.705, -7.035,  4.450,  6.705,
            -8.248, -9.797, -7.907, -4.513, -9.797, -7.907,
        },
        // clang-format on

        // clang-format off
        {
             4.534, -5.908, -9.388,  -INFINITY, -INFINITY, -INFINITY,
             7.889, -5.721,  5.507,  4.295, -INFINITY,  -INFINITY,
             4.534, -5.908, -9.388,  2.356, -5.908, -INFINITY,
             7.889, -5.721,  5.507,  4.295, -5.721,  5.507,

            -7.173,  4.450,  6.705, -INFINITY,  -INFINITY, -INFINITY,
            -8.248, -9.797, -7.907, -4.513,  -INFINITY, -INFINITY,
            -7.173,  4.450,  6.705, -7.035,  4.450,  -INFINITY,
            -8.248, -9.797, -7.907, -4.513, -9.797, -7.907,
        },
    },
};

TEST_P(XAttentionCausalMaskTest, CausalMaskIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.in_shape));

    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD,
                                                            DEFAULT_BLOCK_SIZE,
                                                            DEFAULT_STRIDE);
    std::vector<double> test_out_data = test_struct.in_data;
    selector.apply_causal_mask_(test_out_data.data(), test_struct.in_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-5), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         XAttentionCausalMaskTest,
                         ::testing::ValuesIn(CAUSAL_MASK_TEST_CASES));

struct BlockSumTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    size_t block_size;
    size_t stride;
    ov::Shape out_shape;
    std::vector<double> ref_out_data;
};

using XAttentionBlockSumTest = ::testing::TestWithParam<BlockSumTestData>;

std::vector<BlockSumTestData> BLOCK_SUM_TEST_CASES = {
    {
        {2, 4, 8},
        // clang-format off
        {
            0.1117, 0.0780, 0.1347, 0.0885, 0.1942, 0.0922, 0.1184, 0.1824,
            0.1488, 0.1766, 0.0852, 0.1239, 0.0930, 0.1220, 0.1367, 0.1138,
            0.1410, 0.0861, 0.0774, 0.1325, 0.1478, 0.1689, 0.0885, 0.1579,
            0.1248, 0.1038, 0.1842, 0.0935, 0.1813, 0.0890, 0.0897, 0.1336,

            0.0905, 0.1049, 0.1263, 0.0953, 0.1018, 0.1297, 0.1659, 0.1855,
            0.1373, 0.1791, 0.1005, 0.1286, 0.1492, 0.1373, 0.0820, 0.0860,
            0.0997, 0.1285, 0.0786, 0.1366, 0.1963, 0.0904, 0.1488, 0.1211,
            0.1859, 0.1174, 0.1364, 0.0930, 0.1028, 0.1034, 0.1699, 0.0912
        },
        // clang-format on

        /* block_size = */ 8,
        /* stride = */ 4,
        {2, 2, 4},

        // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
    },
};
TEST_P(XAttentionBlockSumTest, BlockSumIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.out_shape));

    ov::reference::XAttentionBlockSelector<double> selector(DEFAULT_THRESHOLD,
                                                            test_struct.block_size,
                                                            test_struct.stride);
    std::vector<double> test_out_data(test_struct.ref_out_data.size());
    selector.block_sum_attention_scores(test_struct.in_data.data(), test_struct.in_shape, test_out_data.data(), test_struct.out_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-5), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         XAttentionBlockSumTest,
                         ::testing::ValuesIn(BLOCK_SUM_TEST_CASES));

struct BlockSelectTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    double threshold;
    ov::reference::XAttentionRetainedBlockIndicesForAllHeads ref_retained_block_indices;
};

using XAttentionBlockSelectTest = ::testing::TestWithParam<BlockSelectTestData>;

std::vector<BlockSelectTestData> BLOCK_SELECT_TEST_CASES = {
    {
        {2, 2, 5},
        // clang-format off
        {
            0.0000, 0.5151, 0.4323, 0.5014, 0.5513,
            100.0, 0.4557, 0.4876, 0.5870, 0.4697,

            1.7491, 0.3118, 0.4507, 0.5180, 0.5194,
            0.3123, 0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.25,
     {
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}},
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}},
     }},

    {{2, 2, 5},
     // clang-format off
        {
            // larger values in non-causal area should have no impact
            0.4729, 0.5151, 0.4323, 0.5014, 1337.0,
            0.5267, 0.4557, 0.4876, 0.5870, 0.4697,

            0.4647, 0.5118, 0.4507, 0.5180, 42.0,
            0.0000, 0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.45,
     {
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}, {1, 3}},
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}, {1, 1}},
     }},
    {{2, 2, 5},
     // clang-format off
        {
            0.4729, 0.5151, 0.4323, 0.5014, 0.5513,
            0.5267, 0.4557, 0.4876, 0.5870, 0.4697,

            0.4647, 0.5118, 0.4507, 0.5180, 0.5194,
            0.0000, 0.4446, 0.1234, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.8,
     {
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}, {0, 1}, {0, 2}, {1, 3}, {1, 2}},
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}, {0, 1}, {0, 2}, {1, 3}, {1, 1}},
     }},
    {{2, 2, 5},
     // clang-format off
        {
            0.4729, 0.5151, 0.4323, 0.5014, 0.5513,
            0.5267, 0.4557, 0.4876, 0.5870, 0.4697,

            0.4647, 0.5118, 0.4507, 0.5180, 0.5194,
            0.0000, 0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.0,
     {
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}},
         {{0, 0}, {1, 0}, {0, 3}, {1, 4}},
     }},
    {{2, 2, 5},
     // clang-format off
        {
            0.4729, 0.5151, 0.4323, 0.5014, 0.5513,
            0.5267, 0.4557, 0.4876, 0.5870, 0.4697,

            0.4647, 0.5118, 0.4507, 0.5180, 0.5194,
            0.0000, 0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 1.0,
     {
         {{0, 0}, {1, 0}, {1, 3}, {0, 1}, {0, 3}, {1, 2}, {1, 4}, {1, 1}, {0, 2}},
         {{0, 0}, {1, 0}, {1, 1}, {1, 4}, {0, 3}, {0, 1}, {1, 3}, {0, 2}, {1, 2}},
     }},
};

TEST_P(XAttentionBlockSelectTest, BlockSelectionIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));

    ov::reference::XAttentionBlockSelector<double> selector(test_struct.threshold, DEFAULT_BLOCK_SIZE, DEFAULT_STRIDE);
    auto test_result = selector.get_block_indices_to_keep(test_struct.in_data.data(), test_struct.in_shape);

    EXPECT_EQ(test_result, test_struct.ref_retained_block_indices);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, XAttentionBlockSelectTest, ::testing::ValuesIn(BLOCK_SELECT_TEST_CASES));

struct E2EBlockSelectTestData {
    ov::Shape q_shape;
    std::vector<double> q_data;
    ov::Shape k_shape;
    std::vector<double> k_data;
    double threshold;
    size_t block_size;
    size_t stride;
    ov::reference::XAttentionRetainedBlockIndicesForAllHeads ref_retained_block_indices;
};

using XAttentionE2EBlockSelectTest = ::testing::TestWithParam<E2EBlockSelectTestData>;

ov::Shape E2E_Q_SHAPE_8 = {2, 8, 2};
std::vector<double> E2E_Q_DATA_8 = {
    // clang-format off
    -1.2870, -1.2179,  0.0316,  0.0080, -0.6171,  1.0622,  0.3085, -0.7751,
    -1.3612,  0.9485, -0.0803,  0.5752,  0.1925, -0.1113,  1.4693,  0.0673,
     0.7422,  0.7149, -1.7684, -0.0651, -0.1925, -1.4169,  1.0030, -0.8091,
    -0.7934,  0.5160, -0.2543,  0.1729, -0.0687, -1.4245,  0.0758,  1.1613
    // clang-format on
};

ov::Shape E2E_K_SHAPE_8 = {2, 8, 2};
std::vector<double> E2E_K_DATA_8 = {
    // clang-format off
     0.2980,  0.4959, -0.0834,  0.7015,  1.2516,  0.6656, -2.7873,  1.9731,
    -0.4817,  1.1117, -0.8096, -0.5397, -1.0528,  0.2869, -1.1274,  1.4849,
    -0.2468, -1.0449, -1.0085, -0.3389,  0.6750,  0.9095,  0.4674,  2.2321,
     1.3183, -0.3513, -0.3717,  0.0176, -0.2545, -0.6729, -1.1547,  0.0279
    // clang-format on
};

ov::Shape E2E_K_SHAPE_16 = {2, 16, 2};
std::vector<double> E2E_K_DATA_16 = {
    // clang-format off
    -0.9049, -1.9274, -0.3687, -1.1156,  0.1343,  1.1119,  0.7139,  1.0958,
     0.7644,  1.9416,  0.9911,  0.8628,  0.4935, -0.3232, -1.1748,  0.0462,
     0.0488, -0.4271,  1.6657,  0.4596,  1.3253, -1.3023,  0.4961,  1.3707,
    -0.1723, -1.1623, -0.6218, -0.5510,  0.1900,  0.2679, -1.0627,  0.6976,
     0.0737,  0.7033,  1.5972, -0.7547,  0.2586, -0.7601, -0.3851, -0.7056,
    -1.2970, -0.2983,  0.9817,  0.0878,  1.1081, -0.9637,  0.4593, -0.2039,
    -0.3805,  0.1023, -0.2613, -0.5791,  0.2056, -1.1121, -0.0553, -2.4382,
     0.0129, -0.6673, -1.2580, -0.5264,  1.0097, -0.7766,  0.9379,  0.7274
    // clang-format on
};

std::vector<E2EBlockSelectTestData> E2E_BLOCK_SELECT_TEST_CASES = {
    {
        {2, 4, 4},
        // clang-format off
        {
             3.144,  8.512,  8.518, -8.386,
             7.889, -5.721,  5.507,  4.295,
            -6.624, -8.463,  7.474,  9.879,
             4.534, -5.908, -9.388,  2.356,

             7.497,  8.186, -8.658, -4.796,
            -8.248, -9.797, -7.907, -4.513,
             3.469,  7.633,  7.244, -6.844,
            -7.173,  4.450,  6.705, -7.035
        },
        // clang-format on
        {2, 4, 4},
        // clang-format off
        {
             3.144,  8.512,  8.518, -8.386,
             7.889, -5.721,  5.507,  4.295,
            -6.624, -8.463,  7.474,  9.879,
             4.534, -5.908, -9.388,  2.356,

             7.497,  8.186, -8.658, -4.796,
            -8.248, -9.797, -7.907, -4.513,
             3.469,  7.633,  7.244, -6.844,
            -7.173,  4.450,  6.705, -7.035
        },
        // clang-format on

        /* threshold = */ 0.8,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
     {
         {{0, 0}, {1, 1}, {1, 0}},
         {{0, 0}, {1, 1}, {1, 0}}
     }
        // clang-format on
    },

    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 0.0,
        /* block_size = */ 2,
        /* stride = */ 2,
        {{{0, 0}, {0, 4}, {1, 0}, {1, 5}, {2, 0}, {2, 6}, {3, 0}, {3, 7}},
         {{0, 0}, {0, 4}, {1, 0}, {1, 5}, {2, 0}, {2, 6}, {3, 0}, {3, 7}}},
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 1.0,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
        {
            {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}},
            {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 0.8,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
        {

            {{0, 0}, {0, 3}, {0, 4}, {1, 0}, {1, 1}, {1, 3}, {1, 4}, {1, 5}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 5}, {2, 6}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 7}},
            {{0, 0}, {0, 2}, {0, 4}, {1, 0}, {1, 1}, {1, 3}, {1, 5}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 6}, {3, 0}, {3, 1}, {3, 4}, {3, 5}, {3, 6}, {3, 7}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 0.45,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
        {
            {{0, 0}, {0, 4}, {1, 0}, {1, 5}, {2, 0}, {2, 1}, {2, 3}, {2, 6}, {3, 0}, {3, 2}, {3, 5}, {3, 7}},
            {{0, 0}, {0, 2}, {0, 4}, {1, 0}, {1, 5}, {2, 0}, {2, 4}, {2, 6}, {3, 0}, {3, 5}, {3, 7}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 0.45,
        /* block_size = */ 4,
        /* stride = */ 2,

        // clang-format off
        {
            {{0, 0}, {0, 2}, {1, 0}, {1, 1}, {1, 3}},
            {{0, 0}, {0, 2}, {1, 0}, {1, 3}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_16,
        E2E_K_DATA_16,
        /* threshold = */ 0.45,
        /* block_size = */ 4,
        /* stride = */ 4,

        // clang-format off
        {
            {{0, 0}, {0, 2}, {1, 0}, {1, 3}},
            {{0, 0}, {0, 2}, {1, 0}, {1, 3}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_8,
        E2E_K_DATA_8,
        /* threshold = */ 0.5,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
        {
            {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}, {3, 0}, {3, 1}, {3, 3}},
            {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 2}, {3, 0}, {3, 3}}
        }
        // clang-format on
    },
    {
        E2E_Q_SHAPE_8,
        E2E_Q_DATA_8,
        E2E_K_SHAPE_8,
        E2E_K_DATA_8,
        /* threshold = */ 0.2,
        /* block_size = */ 2,
        /* stride = */ 2,

        // clang-format off
        {
            {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 2}, {3, 0}, {3, 3}},
            {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 2}, {3, 0}, {3, 3}}
        }
        // clang-format on
    }};

TEST_P(XAttentionE2EBlockSelectTest, SelectsBlocksCorrectlyFromQKData) {
    auto test_struct = GetParam();
    ov::reference::XAttentionBlockSelector<double> selector(test_struct.threshold,
                                                            test_struct.block_size,
                                                            test_struct.stride);

    auto test_result = selector.select_blocks(test_struct.q_data.data(),
                                              test_struct.q_shape,
                                              test_struct.k_data.data(),
                                              test_struct.k_shape);

    ASSERT_EQ(test_result.size(), test_struct.ref_retained_block_indices.size());
    EXPECT_EQ(test_result, test_struct.ref_retained_block_indices);
    for (size_t head_idx = 0; head_idx < test_result.size(); head_idx++) {
        if (test_result != test_struct.ref_retained_block_indices) {
            std::cout << "Head " << head_idx << std::endl;
            const auto& ref_set = test_struct.ref_retained_block_indices[head_idx];
            const auto& test_set = test_result[head_idx];
            std::cout << "ref has " << ref_set.size() << " elements, test has " << test_set.size() << std::endl;
            std::vector<std::pair<size_t, size_t>> intersection;
            std::set_intersection(ref_set.begin(),
                                  ref_set.end(),
                                  test_set.begin(),
                                  test_set.end(),
                                  std::back_inserter(intersection));

            std::cout << "only ref has ";
            for (const auto& idx : ref_set) {
                if (test_set.find(idx) == test_set.end()) {
                    std::cout << "(" << idx.first << ", " << idx.second << ")" << std::endl;
                }
            }
            std::cout << std::endl;
            std::cout << "only test has ";
            for (const auto& idx : test_set) {
                if (ref_set.find(idx) == ref_set.end()) {
                    std::cout << "(" << idx.first << ", " << idx.second << ")" << std::endl;
                }
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, XAttentionE2EBlockSelectTest, ::testing::ValuesIn(E2E_BLOCK_SELECT_TEST_CASES));
