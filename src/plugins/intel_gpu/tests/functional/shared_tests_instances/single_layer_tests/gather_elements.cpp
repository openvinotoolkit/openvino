// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset6.hpp>

#include "single_layer_tests/gather_elements.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset6;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,
};

INSTANTIATE_TEST_SUITE_P(smoke_set1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 2})),
        ::testing::Values(std::vector<size_t>({2, 2})),
        ::testing::ValuesIn(std::vector<int>({-1, 0, 1})),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 2, 1})),
        ::testing::Values(std::vector<size_t>({4, 2, 1})),
        ::testing::ValuesIn(std::vector<int>({0, -3})),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>({2, 2, 3, 5})),
        ::testing::Values(std::vector<size_t>({2, 2, 3, 7})),
        ::testing::Values(3, -1),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>({3, 2, 3, 8})),
        ::testing::Values(std::vector<size_t>({2, 2, 3, 8})),
        ::testing::Values(0, -4),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_set5, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>({3, 2, 3, 4, 8})),
        ::testing::Values(std::vector<size_t>({3, 2, 3, 5, 8})),
        ::testing::Values(3, -2),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{7, 7, 8, 4}),
        ::testing::Values(std::vector<size_t>{2, 7, 8, 4}),
        ::testing::Values(0),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{6, 1, 8, 4}),
        ::testing::Values(std::vector<size_t>{6, 8, 8, 4}),
        ::testing::Values(1, -3),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{6, 7, 4, 4}),
        ::testing::Values(std::vector<size_t>{6, 7, 2, 4}),
        ::testing::Values(2, -2),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank4axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{6, 5, 8, 7}),
        ::testing::Values(std::vector<size_t>{6, 5, 8, 7}),
        ::testing::Values(3, -1),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 3, 9, 4, 9}),
        ::testing::Values(std::vector<size_t>{1, 3, 9, 4, 9}),
        ::testing::Values(0),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 3, 5, 4, 7}),
        ::testing::Values(std::vector<size_t>{2, 9, 5, 4, 7}),
        ::testing::Values(1, -4),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 2, 6, 8, 9}),
        ::testing::Values(std::vector<size_t>{1, 2, 6, 8, 9}),
        ::testing::Values(2, -3),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 2, 4, 7, 7}),
        ::testing::Values(std::vector<size_t>{2, 2, 4, 3, 7}),
        ::testing::Values(3, -2),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank5axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 3, 9, 3, 2}),
        ::testing::Values(std::vector<size_t>{1, 3, 9, 3, 9}),
        ::testing::Values(4, -1),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis0, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{3, 3, 2, 4, 4, 3}),
        ::testing::Values(std::vector<size_t>{7, 3, 2, 4, 4, 3}),
        ::testing::Values(0),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis1, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 6, 2, 3, 5, 9}),
        ::testing::Values(std::vector<size_t>{1, 6, 2, 3, 5, 9}),
        ::testing::Values(1, -5),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis2, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 3, 9, 7, 2, 1}),
        ::testing::Values(std::vector<size_t>{2, 3, 5, 7, 2, 1}),
        ::testing::Values(2, -4),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis3, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 3, 4, 5, 1, 3}),
        ::testing::Values(std::vector<size_t>{1, 3, 4, 4, 1, 3}),
        ::testing::Values(3, -3),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis4, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{1, 3, 2, 4, 3, 3}),
        ::testing::Values(std::vector<size_t>{1, 3, 2, 4, 6, 3}),
        ::testing::Values(4, -2),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_rank6axis5, GatherElementsLayerTest,
    ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 1, 7, 8, 1, 6}),
        ::testing::Values(std::vector<size_t>{2, 1, 7, 8, 1, 5}),
        ::testing::Values(5, -1),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(idxPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
    GatherElementsLayerTest::getTestCaseName);

}  // namespace
