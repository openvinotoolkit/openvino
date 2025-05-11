// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/roll.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RollLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::f32,
    ov::element::bf16
};

const auto test_case_2D_zero_shifts = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{17, 19}})),  // Input shape
    ::testing::ValuesIn(model_types),                                 // Model type
    ::testing::Values(std::vector<int64_t>{0, 0}),                    // Shift
    ::testing::Values(std::vector<int64_t>{0, 1}),                    // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_1D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({ov::Shape{16}})), // Input shape
    ::testing::ValuesIn(model_types),                            // Model type
    ::testing::Values(std::vector<int64_t>{5}),                  // Shift
    ::testing::Values(std::vector<int64_t>{0}),                  // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_2D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{600, 450}})), // Input shape
    ::testing::ValuesIn(model_types),                                  // Model type
    ::testing::Values(std::vector<int64_t>{300, 250}),                 // Shift
    ::testing::Values(std::vector<int64_t>{0, 1}),                     // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_3D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{2, 320, 320}})), // Input shape
    ::testing::ValuesIn(model_types),                                     // Model type
    ::testing::Values(std::vector<int64_t>{160, 160}),                    // Shift
    ::testing::Values(std::vector<int64_t>{1, 2}),                        // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_negative_unordered_axes_4D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{3, 11, 6, 4}})), // Input shape
    ::testing::ValuesIn(model_types),                                     // Model type
    ::testing::Values(std::vector<int64_t>{7, 3}),                        // Shift
    ::testing::Values(std::vector<int64_t>{-3, -2}),                      // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_repeating_axes_5D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{2, 16, 32, 7, 32}})),  // Input shape
    ::testing::ValuesIn(model_types),                                           // Model type
    ::testing::Values(std::vector<int64_t>{16, 15, 10, 2, 1, 7, 2, 8, 1, 1}),   // Shift
    ::testing::Values(std::vector<int64_t>{-1, -2, -3, 1, 0, 3, 3, 2, -2, -3}), // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_negative_shifts_6D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{4, 16, 3, 6, 5, 2}})), // Input shape
    ::testing::ValuesIn(model_types),                                           // Model type
    ::testing::Values(std::vector<int64_t>{-2, -15, -2, -1, -4, -1}),           // Shift
    ::testing::Values(std::vector<int64_t>{0, 1, 2, 3, 4, 5}),                  // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_case_unord_neg_axes_and_shifts_10D = ::testing::Combine(
    ::testing::Values(
        ov::test::static_shapes_to_test_representation({{2, 2, 4, 2, 3, 6, 3, 2, 3, 2}})), // Input shape
    ::testing::ValuesIn(model_types),                                                      // Model type
    ::testing::Values(std::vector<int64_t>{-2, -1, 1, 1, 1, -2}),                          // Shift
    ::testing::Values(std::vector<int64_t>{-6, -4, -3, 1, -10, -2}),                       // Axes
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_2d_zero_shifts, RollLayerTest,
                            test_case_2D_zero_shifts, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_1d, RollLayerTest,
                            test_case_1D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_2d, RollLayerTest,
                            test_case_2D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_3d, RollLayerTest,
                            test_case_3D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_negative_unordered_axes_4d, RollLayerTest,
                            test_case_negative_unordered_axes_4D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_negative_unordered_axes_5d, RollLayerTest,
                            test_case_repeating_axes_5D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_negative_shifts_6d, RollLayerTest,
                            test_case_negative_shifts_6D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsRoll_unord_neg_shifts_and_axes_10d, RollLayerTest,
                            test_case_unord_neg_axes_and_shifts_10D, RollLayerTest::getTestCaseName);

}  // namespace
