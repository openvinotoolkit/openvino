// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct NonMaxSuppressionParams {
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor maxOutputBoxesPerClass;
    reference_tests::Tensor iouThreshold;
    reference_tests::Tensor scoreThreshold;
    reference_tests::Tensor softNmsSigma;
    op::v5::NonMaxSuppression::BoxEncodingType boxEncoding;
    reference_tests::Tensor expectedSelectedIndices;
    reference_tests::Tensor expectedSelectedScores;
    reference_tests::Tensor expectedValidOutputs;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<NonMaxSuppressionParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, boxes);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, scores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, maxOutputBoxesPerClass);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, iouThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, scoreThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, softNmsSigma);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, boxEncoding);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedSelectedIndices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedSelectedScores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedValidOutputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceNonMaxSuppressionTest : public testing::TestWithParam<NonMaxSuppressionParams>,
                                       public CommonReferenceTest {
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
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        result << "_escType=" << param.expectedSelectedScores.type;
        result << "_escShape=" << param.expectedSelectedScores.shape;
        result << "_evoType=" << param.expectedValidOutputs.type;
        result << "_evoShape=" << param.expectedValidOutputs.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppressionParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Constant>(params.maxOutputBoxesPerClass.type,
                                               params.maxOutputBoxesPerClass.shape,
                                               params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<op::v0::Constant>(params.iouThreshold.type,
                                                                      params.iouThreshold.shape,
                                                                      params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<op::v0::Constant>(params.scoreThreshold.type,
                                                                        params.scoreThreshold.shape,
                                                                        params.scoreThreshold.data.data());
        const auto soft_nms_sigma = std::make_shared<op::v0::Constant>(params.softNmsSigma.type,
                                                                       params.softNmsSigma.shape,
                                                                       params.softNmsSigma.data.data());
        const auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     soft_nms_sigma,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

class ReferenceNonMaxSuppressionTestWithoutConstants : public ReferenceNonMaxSuppressionTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data,
                     params.scores.data,
                     params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data,
                     params.scoreThreshold.data,
                     params.softNmsSigma.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppressionParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Parameter>(params.maxOutputBoxesPerClass.type,
                                                params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold =
            std::make_shared<op::v0::Parameter>(params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold =
            std::make_shared<op::v0::Parameter>(params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto soft_nms_sigma =
            std::make_shared<op::v0::Parameter>(params.softNmsSigma.type, params.softNmsSigma.shape);
        const auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     soft_nms_sigma,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(
            nms->outputs(),
            ParameterVector{boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma});
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
    std::vector<NonMaxSuppressionParams> params{
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                                                         0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                                                         0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CENTER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("nonmaxsuppression_center_point_box_format"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                                                         1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("nonmaxsuppression_flipped_coordinates"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET,
                                            {1, 1, 10},
                                            std::vector<T>{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("nonmaxsuppression_identical_boxes"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("nonmaxsuppression_limit_output_size"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("nonmaxsuppression_single_box"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {2, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                                                                         0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {2, 1, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(
                ET_TH,
                {4, 3},
                std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}))
            .testcaseName("nonmaxsuppression_two_batches"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {1, 2, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
            .expectedSelectedScores(reference_tests::Tensor(
                ET_TH,
                {4, 3},
                std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}))
            .testcaseName("nonmaxsuppression_two_classes"),
    };
    return params;
}

std::vector<NonMaxSuppressionParams> generateCombinedParams() {
    const std::vector<std::vector<NonMaxSuppressionParams>> generatedParams{
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
    std::vector<NonMaxSuppressionParams> params{
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0f, 0.0f,  1.0f, 1.0f,   0.0f, 0.1f,
                                                                         1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                                                                         0.0f, 10.0f, 1.0f, 11.0f,  0.0f, 10.1f,
                                                                         1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {1}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.2f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v5::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0f, 0.0f, 0.95f}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NonMaxSuppressionParams> generateCombinedParamsWithoutConstants() {
    const std::vector<std::vector<NonMaxSuppressionParams>> generatedParams{
        generateParamsWithoutConstants<element::Type_t::bf16,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f16,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f32,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::bf16,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f16,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f32,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppressionParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppressionTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceNonMaxSuppressionTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppressionTestWithoutConstants,
                         testing::ValuesIn(generateCombinedParamsWithoutConstants()),
                         ReferenceNonMaxSuppressionTestWithoutConstants::getTestCaseName);

struct NonMaxSuppression4Params {
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor maxOutputBoxesPerClass;
    reference_tests::Tensor iouThreshold;
    reference_tests::Tensor scoreThreshold;
    op::v4::NonMaxSuppression::BoxEncodingType boxEncoding;
    reference_tests::Tensor expectedSelectedIndices;
    std::string testcaseName;
};

struct Builder4 : ParamsBuilder<NonMaxSuppression4Params> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, boxes);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, scores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, maxOutputBoxesPerClass);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, iouThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, scoreThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, boxEncoding);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, expectedSelectedIndices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder4, testcaseName);
};

class ReferenceNonMaxSuppression4Test : public testing::TestWithParam<NonMaxSuppression4Params>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonMaxSuppression4Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bType=" << param.boxes.type;
        result << "_bShape=" << param.boxes.shape;
        result << "_sType=" << param.scores.type;
        result << "_sShape=" << param.scores.shape;
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression4Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Constant>(params.maxOutputBoxesPerClass.type,
                                               params.maxOutputBoxesPerClass.shape,
                                               params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<op::v0::Constant>(params.iouThreshold.type,
                                                                      params.iouThreshold.shape,
                                                                      params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<op::v0::Constant>(params.scoreThreshold.type,
                                                                        params.scoreThreshold.shape,
                                                                        params.scoreThreshold.data.data());
        const auto nms = std::make_shared<op::v4::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

class ReferenceNonMaxSuppression4TestWithoutConstants : public ReferenceNonMaxSuppression4Test {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data,
                     params.scores.data,
                     params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data,
                     params.scoreThreshold.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression4Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Parameter>(params.maxOutputBoxesPerClass.type,
                                                params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold =
            std::make_shared<op::v0::Parameter>(params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold =
            std::make_shared<op::v0::Parameter>(params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto nms = std::make_shared<op::v4::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(
            nms->outputs(),
            ParameterVector{boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold});
        return f;
    }
};

TEST_P(ReferenceNonMaxSuppression4Test, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNonMaxSuppression4TestWithoutConstants, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression4Params> generateParams4() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression4Params> params{
        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                                                         0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                                                         0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CENTER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_center_point_box_format"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                                                         1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_flipped_coordinates"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET,
                                            {1, 1, 10},
                                            std::vector<T>{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_identical_boxes"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_limit_output_size"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_single_box"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {2, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                                                                         0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {2, 1, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0}))
            .testcaseName("nonmaxsuppression_two_batches"),

        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {1, 2, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
            .testcaseName("nonmaxsuppression_two_classes"),
    };
    return params;
}

std::vector<NonMaxSuppression4Params> generateCombinedParams4() {
    const std::vector<std::vector<NonMaxSuppression4Params>> generatedParams{
        generateParams4<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams4<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams4<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams4<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams4<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams4<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression4Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression4Params> generateParams4WithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression4Params> params{
        Builder4{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0f, 0.0f,  1.0f, 1.0f,   0.0f, 0.1f,
                                                                         1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                                                                         0.0f, 10.0f, 1.0f, 11.0f,  0.0f, 10.1f,
                                                                         1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
            .boxEncoding(op::v4::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NonMaxSuppression4Params> generateCombinedParams4WithoutConstants() {
    const std::vector<std::vector<NonMaxSuppression4Params>> generatedParams{
        generateParams4WithoutConstants<element::Type_t::bf16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams4WithoutConstants<element::Type_t::f16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams4WithoutConstants<element::Type_t::f32,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams4WithoutConstants<element::Type_t::bf16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams4WithoutConstants<element::Type_t::f16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams4WithoutConstants<element::Type_t::f32,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression4Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression4Test,
                         testing::ValuesIn(generateCombinedParams4()),
                         ReferenceNonMaxSuppression4Test::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression4TestWithoutConstants,
                         testing::ValuesIn(generateCombinedParams4WithoutConstants()),
                         ReferenceNonMaxSuppression4TestWithoutConstants::getTestCaseName);

struct NonMaxSuppression3Params {
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor maxOutputBoxesPerClass;
    reference_tests::Tensor iouThreshold;
    reference_tests::Tensor scoreThreshold;
    op::v3::NonMaxSuppression::BoxEncodingType boxEncoding;
    reference_tests::Tensor expectedSelectedIndices;
    std::string testcaseName;
};

struct Builder3 : ParamsBuilder<NonMaxSuppression3Params> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, boxes);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, scores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, maxOutputBoxesPerClass);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, iouThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, scoreThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, boxEncoding);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, expectedSelectedIndices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder3, testcaseName);
};

class ReferenceNonMaxSuppression3Test : public testing::TestWithParam<NonMaxSuppression3Params>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonMaxSuppression3Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bType=" << param.boxes.type;
        result << "_bShape=" << param.boxes.shape;
        result << "_sType=" << param.scores.type;
        result << "_sShape=" << param.scores.shape;
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression3Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Constant>(params.maxOutputBoxesPerClass.type,
                                               params.maxOutputBoxesPerClass.shape,
                                               params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<op::v0::Constant>(params.iouThreshold.type,
                                                                      params.iouThreshold.shape,
                                                                      params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<op::v0::Constant>(params.scoreThreshold.type,
                                                                        params.scoreThreshold.shape,
                                                                        params.scoreThreshold.data.data());
        const auto nms = std::make_shared<op::v3::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

class ReferenceNonMaxSuppression3TestWithoutConstants : public ReferenceNonMaxSuppression3Test {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data,
                     params.scores.data,
                     params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data,
                     params.scoreThreshold.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression3Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Parameter>(params.maxOutputBoxesPerClass.type,
                                                params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold =
            std::make_shared<op::v0::Parameter>(params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold =
            std::make_shared<op::v0::Parameter>(params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto nms = std::make_shared<op::v3::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(
            nms->outputs(),
            ParameterVector{boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold});
        return f;
    }
};

TEST_P(ReferenceNonMaxSuppression3Test, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNonMaxSuppression3TestWithoutConstants, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression3Params> generateParams3() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression3Params> params{
        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                                                         0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                                                         0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CENTER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_center_point_box_format"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                                                         1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_flipped_coordinates"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET,
                                            {1, 1, 10},
                                            std::vector<T>{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_identical_boxes"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_limit_output_size"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_single_box"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores"),

        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {1, 2, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
            .testcaseName("nonmaxsuppression_two_classes"),
    };
    return params;
}

std::vector<NonMaxSuppression3Params> generateCombinedParams3() {
    const std::vector<std::vector<NonMaxSuppression3Params>> generatedParams{
        generateParams3<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams3<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams3<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams3<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams3<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams3<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression3Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression3Params> generateParams3WithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression3Params> params{
        Builder3{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0f, 0.0f,  1.0f, 1.0f,   0.0f, 0.1f,
                                                                         1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                                                                         0.0f, 10.0f, 1.0f, 11.0f,  0.0f, 10.1f,
                                                                         1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
            .boxEncoding(op::v3::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NonMaxSuppression3Params> generateCombinedParams3WithoutConstants() {
    const std::vector<std::vector<NonMaxSuppression3Params>> generatedParams{
        generateParams3WithoutConstants<element::Type_t::bf16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams3WithoutConstants<element::Type_t::f16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams3WithoutConstants<element::Type_t::f32,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i32>(),
        generateParams3WithoutConstants<element::Type_t::bf16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams3WithoutConstants<element::Type_t::f16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams3WithoutConstants<element::Type_t::f32,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression3Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression3Test,
                         testing::ValuesIn(generateCombinedParams3()),
                         ReferenceNonMaxSuppression3Test::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression3TestWithoutConstants,
                         testing::ValuesIn(generateCombinedParams3WithoutConstants()),
                         ReferenceNonMaxSuppression3TestWithoutConstants::getTestCaseName);

struct NonMaxSuppression1Params {
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor maxOutputBoxesPerClass;
    reference_tests::Tensor iouThreshold;
    reference_tests::Tensor scoreThreshold;
    op::v1::NonMaxSuppression::BoxEncodingType boxEncoding;
    reference_tests::Tensor expectedSelectedIndices;
    std::string testcaseName;
};

struct Builder1 : ParamsBuilder<NonMaxSuppression1Params> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, boxes);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, scores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, maxOutputBoxesPerClass);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, iouThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, scoreThreshold);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, boxEncoding);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, expectedSelectedIndices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder1, testcaseName);
};

class ReferenceNonMaxSuppression1Test : public testing::TestWithParam<NonMaxSuppression1Params>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonMaxSuppression1Params>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bType=" << param.boxes.type;
        result << "_bShape=" << param.boxes.shape;
        result << "_sType=" << param.scores.type;
        result << "_sShape=" << param.scores.shape;
        result << "_esiType=" << param.expectedSelectedIndices.type;
        result << "_esiShape=" << param.expectedSelectedIndices.shape;
        if (param.testcaseName != "") {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression1Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Constant>(params.maxOutputBoxesPerClass.type,
                                               params.maxOutputBoxesPerClass.shape,
                                               params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<op::v0::Constant>(params.iouThreshold.type,
                                                                      params.iouThreshold.shape,
                                                                      params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<op::v0::Constant>(params.scoreThreshold.type,
                                                                        params.scoreThreshold.shape,
                                                                        params.scoreThreshold.data.data());
        const auto nms = std::make_shared<op::v1::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

class ReferenceNonMaxSuppression1TestWithoutConstants : public ReferenceNonMaxSuppression1Test {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data,
                     params.scores.data,
                     params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data,
                     params.scoreThreshold.data};
        refOutData = {params.expectedSelectedIndices.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const NonMaxSuppression1Params& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Parameter>(params.maxOutputBoxesPerClass.type,
                                                params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold =
            std::make_shared<op::v0::Parameter>(params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold =
            std::make_shared<op::v0::Parameter>(params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto nms = std::make_shared<op::v1::NonMaxSuppression>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     params.boxEncoding,
                                                                     false);
        const auto f = std::make_shared<Model>(
            nms->outputs(),
            ParameterVector{boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold});
        return f;
    }
};

TEST_P(ReferenceNonMaxSuppression1Test, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNonMaxSuppression1TestWithoutConstants, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression1Params> generateParams1() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression1Params> params{
        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                                                         0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                                                         0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CENTER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_center_point_box_format"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                                                         1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_flipped_coordinates"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                          0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET,
                                            {1, 1, 10},
                                            std::vector<T>{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_identical_boxes"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_limit_output_size"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .testcaseName("nonmaxsuppression_single_box"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores"),

        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                                                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                                                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
            .scores(
                reference_tests::Tensor(ET,
                                        {1, 2, 6},
                                        std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
            .testcaseName("nonmaxsuppression_two_classes"),
    };
    return params;
}

std::vector<NonMaxSuppression1Params> generateCombinedParams1() {
    const std::vector<std::vector<NonMaxSuppression1Params>> generatedParams{
        generateParams1<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams1<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams1<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression1Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NonMaxSuppression1Params> generateParams1WithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NonMaxSuppression1Params> params{
        Builder1{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0f, 0.0f,  1.0f, 1.0f,   0.0f, 0.1f,
                                                                         1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                                                                         0.0f, 10.0f, 1.0f, 11.0f,  0.0f, 10.1f,
                                                                         1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
            .boxEncoding(op::v1::NonMaxSuppression::BoxEncodingType::CORNER)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
            .testcaseName("nonmaxsuppression_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NonMaxSuppression1Params> generateCombinedParams1WithoutConstants() {
    const std::vector<std::vector<NonMaxSuppression1Params>> generatedParams{
        generateParams1WithoutConstants<element::Type_t::bf16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams1WithoutConstants<element::Type_t::f16,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
        generateParams1WithoutConstants<element::Type_t::f32,
                                        element::Type_t::i32,
                                        element::Type_t::f32,
                                        element::Type_t::i64>(),
    };
    std::vector<NonMaxSuppression1Params> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression1Test,
                         testing::ValuesIn(generateCombinedParams1()),
                         ReferenceNonMaxSuppression1Test::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NonMaxSuppression_With_Hardcoded_Refs,
                         ReferenceNonMaxSuppression1TestWithoutConstants,
                         testing::ValuesIn(generateCombinedParams1WithoutConstants()),
                         ReferenceNonMaxSuppression1TestWithoutConstants::getTestCaseName);
}  // namespace
