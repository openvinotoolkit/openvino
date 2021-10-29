// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/non_max_suppression.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace subgraph {

const std::vector<InputShapeParams> inShapeParams = {
    InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{3, 100, 5}}},
    InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{1, 10, 50}}},
    InputShapeParams{std::vector<ov::Dimension>{}, std::vector<TargetShapeParams>{{2, 50, 50}}},
    InputShapeParams{std::vector<ov::Dimension>{-1, -1, -1}, std::vector<TargetShapeParams>{{2, 50, 50}, {3, 100, 5}, {1, 10, 50}}},
    InputShapeParams{std::vector<ov::Dimension>{{1, 5}, {1, 100}, {10, 75}}, std::vector<TargetShapeParams>{{4, 15, 10}, {5, 5, 12}, {1, 35, 15}}}
};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<op::v5::NonMaxSuppression::BoxEncodingType> encodType = {op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                                           op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<element::Type> outType = {element::i32, element::i64};
const std::vector<ngraph::helpers::InputLayerType> maxBoxInputTypes = {ngraph::helpers::InputLayerType::PARAMETER, ngraph::helpers::InputLayerType::CONSTANT};

const auto nmsParams = ::testing::Combine(::testing::ValuesIn(inShapeParams),
                                          ::testing::Combine(::testing::Values(ElementType::f32),
                                                             ::testing::Values(ElementType::i32),
                                                             ::testing::Values(ElementType::f32)),
                                          ::testing::ValuesIn(maxOutBoxPerClass),
                                          ::testing::Combine(::testing::ValuesIn(threshold),
                                                             ::testing::ValuesIn(threshold),
                                                             ::testing::ValuesIn(sigmaThreshold)),
                                          ::testing::ValuesIn(maxBoxInputTypes),
                                          ::testing::ValuesIn(encodType),
                                          ::testing::ValuesIn(sortResDesc),
                                          ::testing::ValuesIn(outType),
                                          ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_NmsLayerTest, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

} // namespace subgraph
} // namespace test
} // namespace ov
