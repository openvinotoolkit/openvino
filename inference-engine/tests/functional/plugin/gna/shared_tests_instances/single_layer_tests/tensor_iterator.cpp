// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/tensor_iterator.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<bool> should_decompose = {false};
    const std::vector<size_t> seqLengths = {1};
    const std::vector<size_t> batches = {1};
    const std::vector<size_t> hiddenSizes = {128, 200, 300};
    const std::vector<size_t> seqAxes = {0, 1};
    const std::vector<float> clip = {0.f};
    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
    };

    INSTANTIATE_TEST_SUITE_P(smoke_TensorIterator, TensorIteratorTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(seqLengths),
                                    ::testing::ValuesIn(batches),
                                    ::testing::ValuesIn(hiddenSizes),
                                    ::testing::ValuesIn(seqAxes),
                                    ::testing::ValuesIn(clip),
                                    ::testing::Values(ngraph::helpers::TensorIteratorBody::LSTM),
                                    ::testing::Values(ngraph::op::RecurrentSequenceDirection::FORWARD),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                            TensorIteratorTest::getTestCaseName);
}  // namespace
