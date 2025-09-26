// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <openvino/reference/xattention.hpp>

double DEFAULT_THRESHOLD = 0.8;
size_t DEFAULT_BLOCK_SIZE = 32;
size_t DEFAULT_STRIDE = 8;

struct E2EBlockSelectTestData {
    ov::Shape q_shape;
    std::vector<double> q_data;
    ov::Shape k_shape;
    std::vector<double> k_data;
    double threshold;
    size_t block_size;
    size_t stride;
};

using XAttentionE2EBlockSelectTest = ::testing::TestWithParam<E2EBlockSelectTestData>;

std::vector<E2EBlockSelectTestData> E2E_BLOCK_SELECT_TEST_CASES = {{
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
}};

TEST_P(XAttentionE2EBlockSelectTest, SelectsBlocksWithoutThrowing) {
    auto test_struct = GetParam();
    ov::reference::XAttentionBlockSelector<double> selector(test_struct.threshold,
                                                            test_struct.block_size,
                                                            test_struct.stride);

    EXPECT_NO_THROW(selector.select_blocks(test_struct.q_data.data(),
                                           test_struct.q_shape,
                                           test_struct.k_data.data(),
                                           test_struct.k_shape));
};

INSTANTIATE_TEST_SUITE_P(VariousInputs, XAttentionE2EBlockSelectTest, ::testing::ValuesIn(E2E_BLOCK_SELECT_TEST_CASES));

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
             4.534, -5.908, -9.388,  2.356, -6.624, -8.463,  7.474,  9.879,
             7.889, -5.721,  5.507,  4.295,  3.144,  8.512,  8.518, -8.386,

            -7.173,  4.450,  6.705, -7.035,  3.469,  7.633,  7.244, -6.844,
            -8.248, -9.797, -7.907, -4.513,  7.497,  8.186, -8.658, -4.796,
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
            -8.410,  6.247,  0.264,  7.095, 1.354, -7.748,
            -7.413,  5.855, -4.142,  2.837, 3.930, -2.122,
             3.664, -2.459,  3.530, -1.083, 1.110, -4.244,

            -9.869, -7.636, -5.892,  7.820, 9.438, -2.421,
             3.568,  8.530, -0.841,  1.935, 1.767,  5.950,
            -5.429,  7.854, -7.414, -3.682, -7.832,  9.163,
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
            -22.45795815, -15.079414324, 20.31074709,
            -17.08315589,  -2.382812673, 2.298610045,

            -5.36919437,  -3.01106803, 19.15517481,
             7.22102005,  23.30210841, 23.79384408
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
        {2, 2, 4},
        // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.25,
     {
         {{1, 2}, {0, 3}},
         {{1, 0}, {1, 3}},
     }},

    {{2, 2, 4},
     // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.35,
     {
         {{1, 2}, {0, 3}, {0, 0}},
         {{1, 0}, {1, 3}, {0, 3}},
     }},
    {{2, 2, 4},
     // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.1,
     {
         {{1, 2}},
         {{1, 0}},
     }},
    {{2, 2, 4},
     // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 0.0,
     {
         {},
         {},
     }},
    {{2, 2, 4},
     // clang-format off
        {
            0.5151, 0.4323, 0.5014, 0.5513,
            0.4557, 0.4876, 0.5870, 0.4697,

            0.5118, 0.4507, 0.5180, 0.5194,
            0.5315, 0.4446, 0.4929, 0.5310
        },
     // clang-format on
     /* threshold = */ 1.0,
     {
         {{1, 2}, {0, 3}, {0, 0}, {0, 2}, {1, 1}, {1, 3}, {1, 0}, {0, 1}},
         {{1, 0}, {1, 3}, {0, 3}, {0, 2}, {0, 0}, {1, 2}, {0, 1}, {1, 1}},
     }},
};

TEST_P(XAttentionBlockSelectTest, BlockSelectionIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));

    ov::reference::XAttentionBlockSelector<double> selector(test_struct.threshold, DEFAULT_THRESHOLD, DEFAULT_STRIDE);
    auto test_result = selector.get_block_indices_to_keep(test_struct.in_data.data(), test_struct.in_shape);

    EXPECT_EQ(test_result, test_struct.ref_retained_block_indices);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, XAttentionBlockSelectTest, ::testing::ValuesIn(BLOCK_SELECT_TEST_CASES));
