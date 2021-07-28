// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_layer/tensor_iterator.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(TensorIteratorTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};
const std::vector<ngraph::helpers::TensorIteratorBody> body = {
    ngraph::helpers::TensorIteratorBody::GRU, ngraph::helpers::TensorIteratorBody::LSTM, ngraph::helpers::TensorIteratorBody::RNN};
const std::vector<bool> decompose = {true, false};
const std::vector<size_t> sequenceLength = {2};
const std::vector<size_t> batch = {1, 10};
const std::vector<size_t> hiddenSize = {128};
const std::vector<size_t> sequenceAxis = {1};
const std::vector<float> clip = {0.f};
const std::vector<ngraph::op::RecurrentSequenceDirection> direction = {
    ngraph::op::RecurrentSequenceDirection::FORWARD, ngraph::op::RecurrentSequenceDirection::REVERSE};

INSTANTIATE_TEST_SUITE_P(smoke_TensorIterator, TensorIteratorTest,
    ::testing::Combine(
        ::testing::ValuesIn(decompose),
        ::testing::ValuesIn(sequenceLength),
        ::testing::ValuesIn(batch),
        ::testing::ValuesIn(hiddenSize),
        ::testing::ValuesIn(sequenceAxis),
        ::testing::ValuesIn(clip),
        ::testing::ValuesIn(body),
        ::testing::ValuesIn(direction),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    TensorIteratorTest::getTestCaseName);
}  // namespace
