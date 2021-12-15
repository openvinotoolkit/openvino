// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/normalize_l2.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace InferenceEngine;
using namespace LayerTestsDefinitions;

namespace {
    TEST_P(NormalizeL2LayerTest, Serialize) {
    Serialize();
}

const std::vector<std::vector<int64_t>> axes = {
        {1},
};
const std::vector<float> eps = { 1e-4f };

const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

const std::vector<Precision> netPrecisions = {
    Precision::FP32,
    Precision::BF16
};

INSTANTIATE_TEST_SUITE_P(smoke_NormalizeL2Serialization, NormalizeL2LayerTest,
        testing::Combine(
            testing::ValuesIn(axes),
            testing::ValuesIn(eps),
            testing::ValuesIn(epsMode),
            testing::Values(std::vector<size_t>{1, 32, 17}),
            testing::ValuesIn(netPrecisions),
            testing::Values(CommonTestUtils::DEVICE_CPU)),
        NormalizeL2LayerTest::getTestCaseName);
}  // namespace
