// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/multiclass_nms.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/benchmark.hpp"

using namespace ov::test::subgraph;
using namespace InferenceEngine;
using namespace ngraph;

/* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
const std::vector<std::vector<ov::Shape>> inStaticShapeParams1 = {
    {{3, 100, 4}, {3,   1, 100}},
    {{1, 10,  4}, {1, 100, 10 }}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams1 = {
    // num_batches, num_boxes, 4
    {{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    // num_batches, num_classes, num_boxes
     {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
        {{1, 3, 10}, {2, 5, 100}}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), ngraph::Dimension(1, 100)}},
        {{1, 3, 10}, {2, 5, 100}}}}
};

/* input format #2 with 3 inputs: bboxes C, M, 4, scores C, M, roisnum N */
const std::vector<std::vector<ov::Shape>> inStaticShapeParams2 = {
    {{1, 10, 4}, {1, 10}, {1}},
    {{1, 10, 4}, {1, 10}, {10}},
    {{2, 100, 4}, {2, 100}, {1}},
    {{2, 100, 4}, {2, 100}, {10}}
};

const std::vector<std::vector<ov::test::InputShape>> inDynamicShapeParams2 = {
    /*0*/
    // bboxes
    {{{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    // scores
     {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
        {{1, 10}, {2, 100}}},
    // roisnum
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*1*/
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100), 4},
        {{1, 10, 4}, {2, 100, 4}}},
    {{{ngraph::Dimension(1, 10), ngraph::Dimension(1, 100)}},
        {{1, 10}, {2, 100}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*2*/
    {{{ngraph::Dimension(2), ngraph::Dimension(100), 4},
        {{2, 100, 4}}},
    {{{ngraph::Dimension(2), ngraph::Dimension(100)}},
        {{2, 100}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}},
    /*3*/
    {{{ngraph::Dimension(3), ngraph::Dimension(2), 4},
        {{3, 2, 4}}},
    {{{ngraph::Dimension(3), ngraph::Dimension(2)}},
        {{3, 2}}},
    {{ngraph::Dimension::dynamic()},
        {{1}, {10}}}}     // more images than num_boxes
};

const std::vector<int32_t> nmsTopK = {-1, 20};
const std::vector<float> iouThreshold = {0.7f};
const std::vector<float> scoreThreshold = {0.7f};
const std::vector<int32_t> backgroundClass = {-1, 1};
const std::vector<int32_t> keepTopK = {-1, 30};
const std::vector<element::Type> outType = {element::i32, element::i64};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> sortResultType = {
    ov::op::util::MulticlassNmsBase::SortResultType::SCORE,
    ov::op::util::MulticlassNmsBase::SortResultType::CLASSID,
    ov::op::util::MulticlassNmsBase::SortResultType::NONE};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<float> nmsEta = {0.6f, 1.0f};
const std::vector<bool> normalized = {true, false};

const auto nmsParamsStatic_smoke1 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams1)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto nmsParamsDynamic_smoke1 = ::testing::Combine(
    ::testing::ValuesIn(inDynamicShapeParams1),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_static1, MulticlassNmsLayerTest, nmsParamsStatic_smoke1, MulticlassNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_dynamic1, MulticlassNmsLayerTest, nmsParamsDynamic_smoke1, MulticlassNmsLayerTest::getTestCaseName);

const auto nmsParamsStatic_smoke2 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inStaticShapeParams2)),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto nmsParamsDynamic_smoke2 = ::testing::Combine(
    ::testing::ValuesIn(inDynamicShapeParams2),
    ::testing::Combine(::testing::Values(ov::element::f32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::i32),
                       ::testing::Values(ov::element::f32)),
    ::testing::ValuesIn(nmsTopK),
    ::testing::Combine(::testing::ValuesIn(iouThreshold), ::testing::ValuesIn(scoreThreshold), ::testing::ValuesIn(nmsEta)),
    ::testing::ValuesIn(backgroundClass),
    ::testing::ValuesIn(keepTopK),
    ::testing::ValuesIn(outType),
    ::testing::ValuesIn(sortResultType),
    ::testing::Combine(::testing::ValuesIn(sortResDesc), ::testing::ValuesIn(normalized)),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_static2, MulticlassNmsLayerTest, nmsParamsStatic_smoke2, MulticlassNmsLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNmsLayerTest_dynamic2, MulticlassNmsLayerTest, nmsParamsDynamic_smoke2, MulticlassNmsLayerTest::getTestCaseName);

namespace {

struct MulticlassNmsBenchmarkTest : ov::test::BenchmarkLayerTest<MulticlassNmsLayerTest> {
    void validate() override {
        MulticlassNmsLayerTest::validate();
    }
};

TEST_P(MulticlassNmsBenchmarkTest, benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run("MulticlassNmsIEInternal", std::chrono::milliseconds(2000), 100);
}

const size_t num_batches = 10;
const size_t num_classes = 100;
const size_t num_boxes = 1000;

const std::vector<std::vector<ov::Shape>> bmInputShapeParams = {
    /* input format #1 with 2 inputs: bboxes N, M, 4, scores N, C, M */
    // {{num_batches, num_boxes, 4}, {num_batches, num_classes, num_boxes}},
    /* input format #2 with 3 inputs: bboxes C, M, 4, scores C, M, roisnum N */
    {{num_batches, num_boxes, 4}, {num_batches, num_classes, num_boxes}},
    {{num_classes, num_boxes, 4}, {num_classes, num_boxes}, {num_batches}}
};

const std::vector<ov::op::util::MulticlassNmsBase::SortResultType> bmSortResultType = {
    ov::op::util::MulticlassNmsBase::SortResultType::SCORE,
    ov::op::util::MulticlassNmsBase::SortResultType::CLASSID,
    ov::op::util::MulticlassNmsBase::SortResultType::NONE
};

const auto multiclassNmsBenchmarkParams = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(bmInputShapeParams)),
    ::testing::Combine(::testing::Values(ov::element::f32),   // input 'boxes' and 'scores' precisions
                       ::testing::Values(ov::element::i32),   // input 'roisnum' precision
                       ::testing::Values(ov::element::i32),   // max_output_boxes_per_class precision
                       ::testing::Values(ov::element::f32)),  // iou_threshold, score_threshold, soft_nms_sigma precisions
    ::testing::Values(-1, 50),                                // nmsTopK, Max output boxes per class
    ::testing::Combine(::testing::Values(0.0f, 0.7f),         // iouThreshold, intersection over union threshold
                       ::testing::Values(0.0f, 0.7f),         // scoreThreshold, minimum score to consider box for the processing
                       ::testing::Values(0.8f, 1.0f)),        // nmsEta, eta parameter for adaptive NMS
    ::testing::Values(-1),                                    // background_class, the background class id, `-1` meaning to keep all classes
    ::testing::Values(-1, 50),                                // keepTopK, maximum number of boxes to be selected per batch element.
    ::testing::Values(element::i64),                          // outType, Output type
    ::testing::ValuesIn(bmSortResultType),                    // SortResultType, sort_result
    ::testing::Combine(::testing::Values(true, false),        // sortResDesc, Sort result across batch
                       ::testing::Values(true)),              // normalized,
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_CASE_P(MulticlassNms_Benchmark,
                        MulticlassNmsBenchmarkTest,
                        multiclassNmsBenchmarkParams,
                        MulticlassNmsBenchmarkTest::getTestCaseName);

}  // namespace
