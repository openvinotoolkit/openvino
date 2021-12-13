// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/constant.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct NonMaxSuppressionParams {
    NonMaxSuppressionParams(
        const Tensor& boxes, const Tensor& scores,
        const Tensor& maxOutputBoxesPerClass, const Tensor& iouThreshold, const Tensor& scoreThreshold,
        const Tensor& softNmsSigma, const op::v5::NonMaxSuppression::BoxEncodingType boxEncoding,
        const Tensor& expectedSelectedIndices, const Tensor& expectedSelectedScores,
        const Tensor& expectedValidOutputs, const std::string& testcaseName = "") :
        boxes(boxes), scores(scores),
        maxOutputBoxesPerClass(maxOutputBoxesPerClass), iouThreshold(iouThreshold), scoreThreshold(scoreThreshold),
        softNmsSigma(softNmsSigma), boxEncoding(boxEncoding),
        expectedSelectedIndices(expectedSelectedIndices), expectedSelectedScores(expectedSelectedScores),
        expectedValidOutputs(expectedValidOutputs), testcaseName(testcaseName) {}

    Tensor boxes;
    Tensor scores;
    Tensor maxOutputBoxesPerClass;
    Tensor iouThreshold;
    Tensor scoreThreshold;
    Tensor softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    Tensor expectedSelectedIndices;
    Tensor expectedSelectedScores;
    Tensor expectedValidOutputs;
    std::string testcaseName;
};

class ReferenceNonMaxSuppressionTest : public testing::TestWithParam<NonMaxSuppressionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonMaxSuppressionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bType=" << param.boxes.type;
        result << "_bShape=" << param.boxes.shape;
        result << "_sType=" << param.scores.type;
        result << "_sShape=" << param.scores.shape;
        result << "_escType=" << param.expectedSelectedScores.type;
        result << "_escShape=" << param.expectedSelectedScores.shape;
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        result << "_evoType=" << param.expectedValidOutputs.type;
        if (param.testcaseName != "") {
            result << "_evoShape=" << param.expectedValidOutputs.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_evoShape=" << param.expectedValidOutputs.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppressionParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class = std::make_shared<op::v0::Constant>(
            params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape, params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<op::v0::Constant>(
            params.iouThreshold.type, params.iouThreshold.shape, params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<op::v0::Constant>(
            params.scoreThreshold.type, params.scoreThreshold.shape, params.scoreThreshold.data.data());
        const auto soft_nms_sigma = std::make_shared<op::v0::Constant>(
            params.softNmsSigma.type, params.softNmsSigma.shape, params.softNmsSigma.data.data());
        const auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     soft_nms_sigma,
                                                                     params.boxEncoding,
                                                                     false);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

class ReferenceNonMaxSuppressionTestWithoutConstants : public ReferenceNonMaxSuppressionTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data, params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data, params.scoreThreshold.data, params.softNmsSigma.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonMaxSuppressionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bType=" << param.boxes.type;
        result << "_bShape=" << param.boxes.shape;
        result << "_sType=" << param.scores.type;
        result << "_sShape=" << param.scores.shape;
        result << "_escType=" << param.expectedSelectedScores.type;
        result << "_escShape=" << param.expectedSelectedScores.shape;
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        result << "_evoType=" << param.expectedValidOutputs.type;
        if (param.testcaseName != "") {
            result << "_evoShape=" << param.expectedValidOutputs.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_evoShape=" << param.expectedValidOutputs.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppressionParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class = std::make_shared<op::v0::Parameter>(
            params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold = std::make_shared<op::v0::Parameter>(
            params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold = std::make_shared<op::v0::Parameter>(
            params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto soft_nms_sigma = std::make_shared<op::v0::Parameter>(
            params.softNmsSigma.type, params.softNmsSigma.shape);
        const auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     soft_nms_sigma,
                                                                     params.boxEncoding,
                                                                     false);
        const auto f = std::make_shared<Model>(nms->outputs(),
                                                  ParameterVector{boxes, scores, max_output_boxes_per_class,
                                                                  iou_threshold, score_threshold, soft_nms_sigma});
        return f;
    }
};

TEST_P(ReferenceNonMaxSuppressionTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNonMaxSuppressionTestWithoutConstants, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppressionParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppressionParams> params {
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
            // selected_indices
            Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}),
            // selected_scores
            Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),
            "nonmaxsuppression_center_point_box_format"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
                0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}),
            // selected_scores
            Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),
            "nonmaxsuppression_flipped_coordinates"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 10, 4}, std::vector<T>{
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}),
            // scores
            Tensor(ET, {1, 1, 10}, std::vector<T>{
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}),
            // selected_scores
            Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{1}),
            "nonmaxsuppression_identical_boxes"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{2}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}),
            // selected_scores
            Tensor(ET_TH, {2, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{2}),
            "nonmaxsuppression_limit_output_size"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}),
            // scores
            Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}),
            // selected_scores
            Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{1}),
            "nonmaxsuppression_single_box"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}),
            // selected_scores
            Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),
            "nonmaxsuppression_suppress_by_IOU"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{3}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {2, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0}),
            // selected_scores
            Tensor(ET_TH, {2, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{2}),
            "nonmaxsuppression_suppress_by_IOU_and_scores"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
            // scores
            Tensor(ET, {2, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{2}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0}),
            // selected_scores
            Tensor(ET_TH, {4, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{4}),
            "nonmaxsuppression_two_batches"),
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
            // scores
            Tensor(ET, {1, 2, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {}, std::vector<T_BOX>{2}),
            // iou_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}),
            // score_threshold
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // soft_nms_sigma
            Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}),
            // selected_scores
            Tensor(ET_TH, {4, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{4}),
            "nonmaxsuppression_two_classes"),
    };
    return params;
}

std::vector<NonMaxSuppressionParams> generateCombinedParams() {
    const std::vector<std::vector<NonMaxSuppressionParams>> generatedParams {
        generateParams<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppressionParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppressionParams> generateParamsWithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppressionParams> params {
        NonMaxSuppressionParams(
            // boxes
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,  1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}),
            // scores
            Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}),
            // max_output_boxes_per_class
            Tensor(ET_BOX, {1}, std::vector<T_BOX>{1}),
            // iou_threshold
            Tensor(ET_TH, {1}, std::vector<T_TH>{0.4f}),
            // score_threshold
            Tensor(ET_TH, {1}, std::vector<T_TH>{0.2f}),
            // soft_nms_sigma
            Tensor(ET_TH, {1}, std::vector<T_TH>{0.0f}),
            // box_encoding
            op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
            // selected_indices
            Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}),
            // selected_scores
            Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0f, 0.0f, 0.95f}),
            // valid_outputs
            Tensor(ET_IND, {1}, std::vector<T_IND>{1}),
            "nonmaxsuppression_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NonMaxSuppressionParams> generateCombinedParamsWithoutConstants() {
    const std::vector<std::vector<NonMaxSuppressionParams>> generatedParams {
        generateParamsWithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppressionParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs, ReferenceNonMaxSuppressionTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceNonMaxSuppressionTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs, ReferenceNonMaxSuppressionTestWithoutConstants,
    testing::ValuesIn(generateCombinedParamsWithoutConstants()), ReferenceNonMaxSuppressionTest::getTestCaseName);
} // namespace
