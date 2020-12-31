// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/op/util/attr_types.hpp>
#include "single_layer_tests/tensor_iterator.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    INSTANTIATE_TEST_CASE_P(smoke_TensorIteratorCommon, TensorIteratorTest,
        ::testing::Combine(
            ::testing::ValuesIn({ false }), // should decompose
            ::testing::ValuesIn(std::vector<size_t>{2}), // seq lengths zero clip
            ::testing::ValuesIn(std::vector<size_t> {20}), // seq lengths clip non zero
            ::testing::ValuesIn(std::vector<size_t> {1, 10}), // hidden size
            ::testing::ValuesIn(std::vector<size_t> {0, 1}), // seq axis
            ::testing::ValuesIn(std::vector<float> {0.f}), // clip
            ::testing::ValuesIn(std::vector<ngraph::helpers::TensorIteratorBody> {ngraph::helpers::TensorIteratorBody::SingleEltwise }), // body type
            ::testing::ValuesIn(std::vector<ngraph::op::RecurrentSequenceDirection>{ngraph::op::RecurrentSequenceDirection::FORWARD,
                                                                                    ngraph::op::RecurrentSequenceDirection::REVERSE }), // direction
            ::testing::ValuesIn(std::vector<InferenceEngine::Precision> {InferenceEngine::Precision::FP32,
                                                                         InferenceEngine::Precision::FP16 }), // precision
            ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        TensorIteratorTest::getTestCaseName);
}  // namespace
