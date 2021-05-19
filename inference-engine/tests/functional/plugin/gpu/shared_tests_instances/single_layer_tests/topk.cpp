// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/topk.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<int64_t> axes = {
        0,
        1,
        2,
};

const std::vector<int64_t> k = {
        1,
        5,
        10,
};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
        ngraph::opset4::TopK::Mode::MIN,
        ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
        ngraph::opset4::TopK::SortType::SORT_INDICES,
        ngraph::opset4::TopK::SortType::SORT_VALUES,
};


INSTANTIATE_TEST_CASE_P(smoke_TopK, TopKLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(k),
                ::testing::ValuesIn(axes),
                ::testing::ValuesIn(modes),
                ::testing::ValuesIn(sortTypes),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({10, 10, 10})),
                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        TopKLayerTest::getTestCaseName);
}  // namespace
