// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/rdft.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RDFTLayerTest;

const std::vector<ov::test::utils::DFTOpType> op_types = {
    ov::test::utils::DFTOpType::FORWARD,
    ov::test::utils::DFTOpType::INVERSE
};

static const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
};

const std::vector<std::vector<size_t>> shapes_forward_1d = {
    {10},
    {64},
    {100},
};


const std::vector<std::vector<int64_t>> signal_sizes_1d = {
    {}, {10},
};

//1D case doesn't work yet on reference implementation
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_RDFT_1d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_forward_1d),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(std::vector<int64_t>{0}),
                            ::testing::ValuesIn(signal_sizes_1d),
                            ::testing::Values(ov::test::utils::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapes_inverse_1d = {
    {10, 2},
    {64, 2},
    {100, 2},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_IRDFT_1d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_inverse_1d),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(std::vector<int64_t>{0}),
                            ::testing::ValuesIn(signal_sizes_1d),
                            ::testing::Values(ov::test::utils::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapes_forward_2d = {
    {10, 15},
    {64, 32},
    {100, 16},
};

const std::vector<std::vector<int64_t>> axes_2d = {
    {0, 1}, {1, 0}, {-2, -1},
};


const std::vector<std::vector<int64_t>> signal_sizes_2d = {
    {}, {10, 10},
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_forward_2d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_2d),
                            ::testing::ValuesIn(signal_sizes_2d),
                            ::testing::Values(ov::test::utils::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapes_inverse_2d = {
    {10, 15, 2},
    {64, 32, 2},
    {100, 32, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_inverse_2d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_2d),
                            ::testing::ValuesIn(signal_sizes_2d),
                            ::testing::Values(ov::test::utils::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapes_forward_4d = {
    {1, 3, 10, 15},
    {1, 4, 64, 32},
};

const std::vector<std::vector<int64_t>> axes_4d = {
    {0, 1, 2, 3}, {1, 0, -2, -1}
};


const std::vector<std::vector<int64_t>> signal_sizes_4d = {
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_forward_4d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_4d),
                            ::testing::ValuesIn(signal_sizes_4d),
                            ::testing::Values(ov::test::utils::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> axes_4d_2d = {
    {2, 3}, {1, -1}
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_axes_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_forward_4d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_4d_2d),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ov::test::utils::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);


const std::vector<std::vector<size_t>> shapes_inverse_4d = {
    {1, 3, 10, 15, 2},
    {1, 4, 64, 32, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_4d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_inverse_4d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_4d),
                            ::testing::ValuesIn(signal_sizes_4d),
                            ::testing::Values(ov::test::utils::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_4d_axes_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapes_inverse_4d),
                            ::testing::ValuesIn(model_types),
                            ::testing::ValuesIn(axes_4d_2d),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ov::test::utils::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);



