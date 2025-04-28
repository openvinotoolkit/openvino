// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/nms_topk.hpp"

namespace ov {
namespace test {

namespace {

const ov::Shape inputShape = {1, 3614, 4};

const std::vector<ov::element::Type> inputPrecisions = {ElementType::f32, ElementType::f16, ElementType::bf16};

const int64_t maxOutputBoxesPerClass = {200};

const float iouThreshold = {0.1f};

const std::vector<float> scoreThresholds = {0.8f, 0.9f};

INSTANTIATE_TEST_SUITE_P(smoke_NMSTopK,
                         NMSTopKTest,
                         ::testing::Combine(::testing::Values(inputShape),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(maxOutputBoxesPerClass),
                                            ::testing::Values(iouThreshold),
                                            ::testing::ValuesIn(scoreThresholds),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         NMSTopKTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
