// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/rdft.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::helpers::DFTOpType> opTypes = {
    ngraph::helpers::DFTOpType::FORWARD,
    ngraph::helpers::DFTOpType::INVERSE
};

static const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> shapesForward1d = {
    {10},
    {64},
    {100},
};


const std::vector<std::vector<int64_t>> signalSizes1d = {
    {}, {10},
};

//1D case doesn't work yet on reference implementation
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_RDFT_1d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesForward1d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::Values(std::vector<int64_t>{0}),
                            ::testing::ValuesIn(signalSizes1d),
                            ::testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesInverse1d = {
    {10, 2},
    {64, 2},
    {100, 2},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_IRDFT_1d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesInverse1d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::Values(std::vector<int64_t>{0}),
                            ::testing::ValuesIn(signalSizes1d),
                            ::testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesForward2d = {
    {10, 15},
    {64, 32},
    {100, 16},
};

const std::vector<std::vector<int64_t>> axes2d = {
    {0, 1}, {1, 0}, {-2, -1},
};


const std::vector<std::vector<int64_t>> signalSizes2d = {
    {}, {10, 10},
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesForward2d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes2d),
                            ::testing::ValuesIn(signalSizes2d),
                            ::testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesInverse2d = {
    {10, 15, 2},
    {64, 32, 2},
    {100, 32, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesInverse2d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes2d),
                            ::testing::ValuesIn(signalSizes2d),
                            ::testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> shapesForward4d = {
    {1, 3, 10, 15},
    {1, 4, 64, 32},
};

const std::vector<std::vector<int64_t>> axes4d = {
    {0, 1, 2, 3}, {1, 0, -2, -1}
};


const std::vector<std::vector<int64_t>> signalSizes4d = {
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesForward4d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes4d),
                            ::testing::ValuesIn(signalSizes4d),
                            ::testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> axes4d_2d = {
    {2, 3}, {1, -1}
};

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_axes_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesForward4d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes4d_2d),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);


const std::vector<std::vector<size_t>> shapesInverse4d = {
    {1, 3, 10, 15, 2},
    {1, 4, 64, 32, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_4d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesInverse4d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes4d),
                            ::testing::ValuesIn(signalSizes4d),
                            ::testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_IRDFT_4d_axes_2d, RDFTLayerTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(shapesInverse4d),
                            ::testing::ValuesIn(inputPrecision),
                            ::testing::ValuesIn(axes4d_2d),
                            ::testing::Values(std::vector<int64_t>{}),
                            ::testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)), RDFTLayerTest::getTestCaseName);



