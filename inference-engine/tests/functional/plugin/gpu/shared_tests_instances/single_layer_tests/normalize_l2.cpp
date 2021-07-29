// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/normalize_l2.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<int64_t>> axes = {
        {},
        {1},
};
const std::vector<float> eps = {1e-7f, 1e-6f, 1e-5f, 1e-4f};

const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

const auto normL2params = testing::Combine(
        testing::ValuesIn(axes),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsMode),
        testing::Values(std::vector<size_t>{1, 3, 10, 5}),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2,
        NormalizeL2LayerTest,
        normL2params,
        NormalizeL2LayerTest::getTestCaseName
);
}  // namespace
