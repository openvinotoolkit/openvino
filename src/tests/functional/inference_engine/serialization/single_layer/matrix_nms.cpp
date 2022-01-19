// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/matrix_nms.hpp"

using namespace ngraph;
using namespace ov::test::subgraph;

namespace {
    TEST_P(MatrixNmsLayerTest, Serialize) {
        serialize();
    }

    const std::vector<ov::test::ElementType> netPrecisions = {
            ov::element::f32,
            ov::element::f16
    };

    const std::vector<std::vector<ov::test::InputShape>> shapeParams = {
        // num_batches, num_boxes, 4
        {{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 4},
            {{1, 10, 4}, {2, 100, 4}}},
        // num_batches, num_classes, num_boxes
        {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
            {{1, 3, 10}, {2, 5, 100}}}},
        // num_batches, num_boxes, 4
        {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), 4},
            {{1, 10, 4}, {2, 100, 4}}},
        // num_batches, num_classes, num_boxes
        {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), ngraph::Dimension(1, 100)}},
            {{1, 3, 10}, {2, 5, 100}}}}
    };

    const std::vector<op::v8::MatrixNms::SortResultType> sortResultType = {op::v8::MatrixNms::SortResultType::CLASSID,
                                                                       op::v8::MatrixNms::SortResultType::SCORE,
                                                                       op::v8::MatrixNms::SortResultType::NONE};
    const std::vector<element::Type> outType = {element::i32, element::i64};
    const std::vector<TopKParams> topKParams = {
        TopKParams{-1, 5},
        TopKParams{100, -1}
    };
    const std::vector<ThresholdParams> thresholdParams = {
        ThresholdParams{0.0f, 2.0f, 0.0f},
        ThresholdParams{0.1f, 1.5f, 0.2f}
    };
    const std::vector<int> nmsTopK = {-1, 100};
    const std::vector<int> keepTopK = {-1, 5};
    const std::vector<int> backgroudClass = {-1, 0};
    const std::vector<bool> normalized = {true, false};
    const std::vector<op::v8::MatrixNms::DecayFunction> decayFunction = {op::v8::MatrixNms::DecayFunction::GAUSSIAN,
                                                    op::v8::MatrixNms::DecayFunction::LINEAR};
    const auto nmsParams = ::testing::Combine(::testing::ValuesIn(shapeParams),
                                          ::testing::Combine(::testing::Values(ov::element::f32),
                                                             ::testing::Values(ov::element::i32),
                                                             ::testing::Values(ov::element::f32)),
                                          ::testing::ValuesIn(sortResultType),
                                          ::testing::ValuesIn(outType),
                                          ::testing::ValuesIn(topKParams),
                                          ::testing::ValuesIn(thresholdParams),
                                          ::testing::ValuesIn(backgroudClass),
                                          ::testing::ValuesIn(normalized),
                                          ::testing::ValuesIn(decayFunction),
                                          ::testing::Values(CommonTestUtils::DEVICE_CPU));

    INSTANTIATE_TEST_SUITE_P(smoke_MatrixNmsLayerTest, MatrixNmsLayerTest, nmsParams, MatrixNmsLayerTest::getTestCaseName);
}  // namespace
