// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <openvino/reference/adaptive_rkv_diversity.hpp>

namespace adaptive_rkv_test {
size_t DEFAULT_BLOCK_SIZE = 2;
size_t DEFAULT_START_SIZE = 2;
size_t DEFAULT_EVICTION_SIZE = 10;

TEST(AdaptiveRKVE2ESmokeTest, CalculatesDiversityWithoutThrowing) {
    ov::reference::AdaptiveRKVDiversityCalculator<double> calculator(DEFAULT_START_SIZE,
                                                                     DEFAULT_EVICTION_SIZE,
                                                                     DEFAULT_BLOCK_SIZE);

    ov::Shape mock_shape{2, (DEFAULT_START_SIZE + DEFAULT_EVICTION_SIZE) * 2, 8};
    std::vector<double> mock_data(ov::shape_size(mock_shape), 1.0);

    EXPECT_NO_THROW(calculator.calculate_block_diversity(mock_data.data(), mock_shape));
};

struct FillDiagonalTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    std::vector<double> ref_out_data;
};

using AdaptiveRKVDiversityFillDiagonalTest = ::testing::TestWithParam<FillDiagonalTestData>;

std::vector<FillDiagonalTestData> FILL_DIAGONAL_TEST_CASES = {{
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

    // clang-format off
        {
             42.00,  8.512,  8.518, -8.386,
             7.889,  42.00,  5.507,  4.295,
            -6.624, -8.463,  42.00,  9.879,
             4.534, -5.908, -9.388,  42.00,

             42.00,  8.186, -8.658, -4.796,
            -8.248,  42.00, -7.907, -4.513,
             3.469,  7.633,  42.00, -6.844,
            -7.173,  4.450,  6.705,  42.00
        },
    // clang-format on
}};

TEST_P(AdaptiveRKVDiversityFillDiagonalTest, FillsDiagonal) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.in_shape));

    ov::reference::AdaptiveRKVDiversityCalculator<double> calculator(DEFAULT_START_SIZE,
                                                                     DEFAULT_EVICTION_SIZE,
                                                                     DEFAULT_BLOCK_SIZE);

    std::vector<double> test_out_data = test_struct.in_data;
    calculator.fill_diagonal_(test_out_data.data(), test_struct.in_shape, 42.0);
    EXPECT_EQ(test_out_data, test_struct.ref_out_data);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         AdaptiveRKVDiversityFillDiagonalTest,
                         ::testing::ValuesIn(FILL_DIAGONAL_TEST_CASES));

struct FillLowValuesWithZerosTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    ov::Shape means_shape;
    std::vector<double> means;
    std::vector<double> ref_out_data;
};

using AdaptiveRKVFillLowValuesWithZerosTest = ::testing::TestWithParam<FillLowValuesWithZerosTestData>;

std::vector<FillLowValuesWithZerosTestData> FILL_LOW_VALUES_WITH_ZEROS_TEST_CASES = {
    {
        {2, 4, 4},
        // clang-format off
        {
             4.534, -5.908, -9.388,  2.356,
            -6.624, -8.463,  7.474,  9.879,
             7.889, -5.721,  5.507,  4.295,
             3.144,  8.512,  8.518, -8.386,

            -7.173,  4.450,  6.705, -7.035,
             3.469,  7.633,  7.244, -6.844,
            -8.248, -9.797, -7.907, -4.513,
             7.497,  8.186, -8.658, -4.796,
        },
        // clang-format on

        {2, 4},

        // clang-format off
        {
            -2.1015,
            0.5665,
            2.9925,
            2.947,

            -0.76325,
            2.8755,
            -7.61625,
            0.55725
        },

        // clang-format off
        {
             4.534,  0.000,  0.000,  2.356,
             0.000,  0.000,  7.474,  9.879,
             7.889,  0.000,  5.507,  4.295,
             3.144,  8.512,  8.518,  0.000,

             0.000,  4.450,  6.705,  0.000,
             3.469,  7.633,  7.244,  0.000,
             0.000,  0.000,  0.000, -4.513,
             7.497,  8.186,  0.000,  0.000,
        },
        // clang-format on
    },
};

TEST_P(AdaptiveRKVFillLowValuesWithZerosTest, FillsLowValuesWithZero) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.means.size(), ov::shape_size(test_struct.means_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.in_shape));

    ov::reference::AdaptiveRKVDiversityCalculator<double> calculator(DEFAULT_START_SIZE,
                                                                     DEFAULT_EVICTION_SIZE,
                                                                     DEFAULT_BLOCK_SIZE);
    std::vector<double> test_out_data = test_struct.in_data;
    calculator.fill_low_values_with_zeros_(test_out_data.data(),
                                           test_struct.in_shape,
                                           test_struct.means.data(),
                                           test_struct.means_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-8), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         AdaptiveRKVFillLowValuesWithZerosTest,
                         ::testing::ValuesIn(FILL_LOW_VALUES_WITH_ZEROS_TEST_CASES));

struct BlockSumTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    size_t block_size;
    ov::Shape out_shape;
    std::vector<double> ref_out_data;
};

using AdaptiveRKVBlockSumTest = ::testing::TestWithParam<BlockSumTestData>;

std::vector<BlockSumTestData> BLOCK_SUM_TEST_CASES = {
    {
        {8, 8},
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

        /* block_size = */ 2,

        {4, 8},

        // clang-format off
        {
            -0.2605, -0.2546, -0.2199, -0.2124, -0.2872, -0.2142, -0.2551, -0.2962,
            -0.2658, -0.1899, -0.2616, -0.226,  -0.3291, -0.2579, -0.1782, -0.2915,
            -0.2278, -0.284 , -0.2268, -0.2239, -0.251,  -0.267,  -0.2479, -0.2715,
            -0.2856, -0.2459, -0.215,  -0.2296, -0.2991, -0.1938, -0.3187, -0.2123

        },
    },
};

TEST_P(AdaptiveRKVBlockSumTest, BlockSumIsCorrect) {
    auto test_struct = GetParam();
    ASSERT_EQ(test_struct.in_data.size(), ov::shape_size(test_struct.in_shape));
    ASSERT_EQ(test_struct.ref_out_data.size(), ov::shape_size(test_struct.out_shape));

    ov::reference::AdaptiveRKVDiversityCalculator<double> calculator(DEFAULT_START_SIZE,
                                                                     DEFAULT_EVICTION_SIZE,
                                                                     test_struct.block_size);
    std::vector<double> test_out_data(test_struct.ref_out_data.size());
    calculator.block_sum_diversity_values(test_struct.in_data.data(), test_struct.in_shape, test_out_data.data(), test_struct.out_shape);

    EXPECT_THAT(test_out_data, ::testing::Pointwise(::testing::DoubleNear(1e-5), test_struct.ref_out_data));
}

INSTANTIATE_TEST_SUITE_P(VariousInputs,
                         AdaptiveRKVBlockSumTest,
                         ::testing::ValuesIn(BLOCK_SUM_TEST_CASES));

struct DiversityCalculateTestData {
    ov::Shape in_shape;
    std::vector<double> in_data;
    double threshold;

};

struct E2EDiversityTestData {
    ov::Shape k_shape;
    std::vector<double> k_data;
    size_t start_size;
    size_t eviction_size;
    std::vector<std::vector<double>> ref_diversity_data;
};

using AdaptiveRKVE2EDiversityTest = ::testing::TestWithParam<E2EDiversityTestData>;

std::vector<E2EDiversityTestData> E2E_DIVERSITY_TEST_CASES = {
    // basic
    {
        {1, 4, 1},
        // clang-format off
        {
           1.0,
           1.0,
           1.0,
           1.0
        },
        /* start_size = */ 2,
        /* eviction_size = */ 2,
        {{-1.0, -1.0}}
    },
    // larger basic
    {
        {1, 6, 1},
        // clang-format off
        {
           6.5,
           -11.0,
           1.0,
           1.0,
           1.0,
           1.0,
        },
        /* start_size = */ 2,
        /* eviction_size = */ 4,
        {{-1.0, -1.0, -2.0, -2.0},
         {-2.0, -2.0, -1.0, -1.0}}
    },
    // two heads basic
    {
        {2, 8, 1},
        // clang-format off
        {
           6.5,
           -11.0,
           1.0,
           1.0,
           1.0,
           1.0,
           42.0,
           -13.7,

            1337.0,
            -1256.9,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.2,
            0.0
        },
        /* start_size = */ 2,
        /* eviction_size = */ 4,
        {{-1.0, -1.0, -2.0, -2.0},
         {-2.0, -2.0, -1.0, -1.0}}
    },
    // zeroed second head (where it matters)
    {
        {2, 8, 1},
        // clang-format off
        {
           6.5,
           -11.0,
           1.0,
           1.0,
           1.0,
           1.0,
           42.0,
           -13.7,

            1337.0,
            -1256.9,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0
        },
        /* start_size = */ 2,
        /* eviction_size = */ 4,
        {{-0.5, -0.5, -1.0, -1.0},
         {-1.0, -1.0, -0.5, -0.5}}
    },
    // more embedding dimensions
    {
        {2, 8, 4},
        // clang-format off
        {
           6.5, 8.3, 5.1, -7.4,
           -11.0, 1.9, 7.1, 4.8,
           8.0, 8.0, 8.0, 8.0,
           8.0, 8.0, 8.0, 8.0,
           8.0, 8.0, 8.0, 8.0,
           8.0, 8.0, 8.0, 8.0,
           42.0, -41.7, 8.3, 1.0,
           -13.7, 0.0, 0.0, 15.1,

            1337.0, -1.9, -1.4, 475.1,
            -1256.9, 1.0, 789.0, 1421.3,
            -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0,
            0.2, -81.3, 74.3, -641.1,
            0.0, 14.7, 98.1, -27.7
        },
        /* start_size = */ 2,
        /* eviction_size = */ 4,
        {{-1.0, -1.0, -2.0, -2.0},
         {-2.0, -2.0, -1.0, -1.0}}
    },
    // orthogonal tokens
    {
        {2, 8, 4},
        // clang-format off
        {
           6.5, 8.3, 5.1, -7.4,
           -11.0, 1.9, 7.1, 4.8,
           8.0,    0.0,   0.0, 0.0,
           0.0,    0.0, -18.0, 0.0,
           0.0,    0.0,   0.0, 0.1,
           0.0, 1288.0,   0.0, 0.0,
           42.0, -41.7, 8.3, 1.0,
           -13.7, 0.0, 0.0, 15.1,

            1337.0, -1.9, -1.4, 475.1,
            -1256.9, 1.0, 789.0, 1421.3,
            0.0,   0.0,  2.0,  0.0,
            0.0, -12.0,  0.0,  0.0,
            12.8,  0.0,  0.0,  0.0,
            0.0,   0.0,  0.0, 65.5,
            0.2, -81.3, 74.3, -641.1,
            0.0, 14.7, 98.1, -27.7
        },
        /* start_size = */ 2,
        /* eviction_size = */ 4,
        {{0.0, 0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0, 0.0}}
    },
    // random excel-checked golden
    {
        {2, 10, 4},
        // clang-format off
        {
              4.949, -7.294, -6.330,  3.757,
             -3.561,  1.029,  5.030, -9.483,
              5.350, -2.745, -1.404, -7.788,
             -1.086,  4.576, -8.726, -8.815,
              3.144,  8.512,  8.518, -8.386,
              7.889, -5.721,  5.507,  4.295,
             -6.624, -8.463,  7.474,  9.879,
              4.534, -5.908, -9.388,  2.356,
              7.497,  8.186, -8.658, -4.796,
             -8.248, -9.797, -7.907, -4.513,

              3.469,  7.633,  7.244, -6.844,
             -7.173,  4.450,  6.705, -7.035,
              8.773, -7.571, -9.878, -9.584,
              0.807,  8.059, -7.172,  4.303,
             -3.323, -8.852,  1.167, -1.126,
             -4.428,  9.678, -6.547,  0.037,
             -8.152, -9.865,  3.694, -7.650,
              0.359,  8.018, -7.152, -6.242,
             -9.120, -7.228, -9.186,  3.202,
             -9.304, -0.401, -5.287,  6.834
        },
        // clang-format on

        /* start_size = */ 2,
        /* eviction_size = */ 6,
        {{-0.237145, -0.237145, -0.352696, -0.487902, -0.072365, -0.707192},
         {-0.334657, -0.505941, 0, 0.036135, -0.634881, -0.490221},
         {-0.380811, -0.398746801, -0.432080003, -0.693021748, 0, 0.067216441}},
    }};

TEST_P(AdaptiveRKVE2EDiversityTest, CalculatesDiversityCorrectly) {
    auto test_struct = GetParam();
    ov::reference::AdaptiveRKVDiversityCalculator<double> calculator(test_struct.start_size,
                                                                     test_struct.eviction_size,
                                                                     DEFAULT_BLOCK_SIZE);

    auto test_diversity = calculator.calculate_block_diversity(test_struct.k_data.data(), test_struct.k_shape);
    ASSERT_EQ(test_diversity.size(), test_struct.ref_diversity_data.size());
    for (size_t i = 0; i < test_diversity.size(); i++) {
        ASSERT_EQ(test_diversity[i].size(), test_struct.ref_diversity_data[i].size());
    }

    for (size_t i = 0; i < test_diversity.size(); i++) {
        EXPECT_THAT(test_diversity[i],
                    ::testing::Pointwise(::testing::DoubleNear(1e-6), test_struct.ref_diversity_data[i]));
    }
};

INSTANTIATE_TEST_SUITE_P(VariousInputs, AdaptiveRKVE2EDiversityTest, ::testing::ValuesIn(E2E_DIVERSITY_TEST_CASES));
}  // namespace adaptive_rkv_test
