// Copyright (C) 2018-2023 Intel Corporation
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

const std::vector<ov::op::TopKMode> modes = {
        ov::op::TopKMode::MIN,
        ov::op::TopKMode::MAX
};

const std::vector<ov::op::TopKSortType> sortTypes = {
        ov::op::TopKSortType::SORT_INDICES,
        ov::op::TopKSortType::SORT_VALUES,
};

const std::vector<bool> stable = {
        false,
        true,
};


INSTANTIATE_TEST_SUITE_P(smoke_TopK, TopKLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(k),
                ::testing::ValuesIn(axes),
                ::testing::ValuesIn(modes),
                ::testing::ValuesIn(sortTypes),
                ::testing::ValuesIn(stable),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({10, 10, 10})),
                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
        TopKLayerTest::getTestCaseName);
}  // namespace
