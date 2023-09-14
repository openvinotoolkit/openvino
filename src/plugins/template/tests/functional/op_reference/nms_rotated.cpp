// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset13.hpp"
#include "openvino/opsets/opset1.hpp"

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct NMSRotatedParams {
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor maxOutputBoxesPerClass;
    reference_tests::Tensor iouThreshold;
    reference_tests::Tensor scoreThreshold;
    reference_tests::Tensor softNmsSigma;
    opset13::NMSRotated::BoxEncodingType boxEncoding;
    reference_tests::Tensor expectedSelectedIndices;
    reference_tests::Tensor expectedSelectedScores;
    reference_tests::Tensor expectedValidOutputs;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<NMSRotatedParams> {
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

class ReferenceNMSRotatedTest : public testing::TestWithParam<NMSRotatedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NMSRotatedParams>& obj) {
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
    static std::shared_ptr<Model> CreateFunction(const NMSRotatedParams& params) {
        const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class = std::make_shared<opset1::Constant>(
            params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape, params.maxOutputBoxesPerClass.data.data());
        const auto iou_threshold = std::make_shared<opset1::Constant>(
            params.iouThreshold.type, params.iouThreshold.shape, params.iouThreshold.data.data());
        const auto score_threshold = std::make_shared<opset1::Constant>(
            params.scoreThreshold.type, params.scoreThreshold.shape, params.scoreThreshold.data.data());
        const auto soft_nms_sigma = std::make_shared<opset1::Constant>(
            params.softNmsSigma.type, params.softNmsSigma.shape, params.softNmsSigma.data.data());
        const auto nms = std::make_shared<opset13::NMSRotated>(boxes,
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

class ReferenceNMSRotatedTestWithoutConstants : public ReferenceNMSRotatedTest {
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

private:
    static std::shared_ptr<Model> CreateFunction(const NMSRotatedParams& params) {
        const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class = std::make_shared<opset1::Parameter>(
            params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold = std::make_shared<opset1::Parameter>(
            params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold = std::make_shared<opset1::Parameter>(
            params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto soft_nms_sigma = std::make_shared<opset1::Parameter>(
            params.softNmsSigma.type, params.softNmsSigma.shape);
        const auto nms = std::make_shared<opset13::NMSRotated>(boxes,
                                                                     scores,
                                                                     max_output_boxes_per_class,
                                                                     iou_threshold,
                                                                     score_threshold,
                                                                     soft_nms_sigma,
                                                                     params.boxEncoding,
                                                                     false,
                                                                     params.expectedSelectedIndices.type);
        const auto f = std::make_shared<Model>(nms->outputs(),
                                               ParameterVector{boxes, scores, max_output_boxes_per_class,
                                                               iou_threshold, score_threshold, soft_nms_sigma});
        return f;
    }
};

TEST_P(ReferenceNMSRotatedTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNMSRotatedTestWithoutConstants, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NMSRotatedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NMSRotatedParams> params {
        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CENTER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
        .testcaseName(
            "NMSRotated_center_point_box_format"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
                0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
        .testcaseName(
            "NMSRotated_flipped_coordinates"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{
                0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
        .testcaseName(
            "NMSRotated_identical_boxes"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
        .testcaseName(
            "NMSRotated_limit_output_size"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
        .testcaseName(
            "NMSRotated_single_box"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 0, 5}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
        .testcaseName(
            "NMSRotated_suppress_by_IOU"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
        .testcaseName(
            "NMSRotated_suppress_by_IOU_and_scores"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
        .scores(
            reference_tests::Tensor(ET, {2, 1, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {4, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}))
        .testcaseName(
            "NMSRotated_two_batches"),

        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
        .scores(
            reference_tests::Tensor(ET, {1, 2, 6}, std::vector<T>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
                0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {4, 3}, std::vector<T_TH>{
                0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}))
        .testcaseName(
            "NMSRotated_two_classes"),
    };
    return params;
}

std::vector<NMSRotatedParams> generateCombinedParams() {
    const std::vector<std::vector<NMSRotatedParams>> generatedParams {
        generateParams<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NMSRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NMSRotatedParams> generateParamsWithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NMSRotatedParams> params {
        Builder {}
        .boxes(
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,  1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
                0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
        .scores(
            reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
                0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
        .maxOutputBoxesPerClass(
            reference_tests::Tensor(ET_BOX, {1}, std::vector<T_BOX>{1}))
        .iouThreshold(
            reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.4f}))
        .scoreThreshold(
            reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.2f}))
        .softNmsSigma(
            reference_tests::Tensor(ET_TH, {1}, std::vector<T_TH>{0.0f}))
        .boxEncoding(
            opset13::NMSRotated::BoxEncodingType::CORNER)
        .expectedSelectedIndices(
            reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
        .expectedSelectedScores(
            reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0f, 0.0f, 0.95f}))
        .expectedValidOutputs(
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
        .testcaseName(
            "NMSRotated_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NMSRotatedParams> generateCombinedParamsWithoutConstants() {
    const std::vector<std::vector<NMSRotatedParams>> generatedParams {
        generateParamsWithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NMSRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotatedTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceNMSRotatedTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotatedTestWithoutConstants,
    testing::ValuesIn(generateCombinedParamsWithoutConstants()), ReferenceNMSRotatedTestWithoutConstants::getTestCaseName);


// struct NMSRotated4Params {
//     reference_tests::Tensor boxes;
//     reference_tests::Tensor scores;
//     reference_tests::Tensor maxOutputBoxesPerClass;
//     reference_tests::Tensor iouThreshold;
//     reference_tests::Tensor scoreThreshold;
//     opset4::NMSRotated::BoxEncodingType boxEncoding;
//     reference_tests::Tensor expectedSelectedIndices;
//     std::string testcaseName;
// };

// struct Builder4 : ParamsBuilder<NMSRotated4Params> {
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, boxes);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, scores);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, maxOutputBoxesPerClass);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, iouThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, scoreThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, boxEncoding);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, expectedSelectedIndices);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder4, testcaseName);
// };

// class ReferenceNMSRotated4Test : public testing::TestWithParam<NMSRotated4Params>, public CommonReferenceTest {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

//     static std::string getTestCaseName(const testing::TestParamInfo<NMSRotated4Params>& obj) {
//         auto param = obj.param;
//         std::ostringstream result;
//         result << "bType=" << param.boxes.type;
//         result << "_bShape=" << param.boxes.shape;
//         result << "_sType=" << param.scores.type;
//         result << "_sShape=" << param.scores.shape;
//         result << "_esiType=" << param.expectedSelectedIndices.type;
//         result << "_esiShape=" << param.expectedSelectedIndices.shape;
//         if (param.testcaseName != "") {
//             result << "_=" << param.testcaseName;
//         }
//         return result.str();
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated4Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Constant>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape, params.maxOutputBoxesPerClass.data.data());
//         const auto iou_threshold = std::make_shared<opset1::Constant>(
//             params.iouThreshold.type, params.iouThreshold.shape, params.iouThreshold.data.data());
//         const auto score_threshold = std::make_shared<opset1::Constant>(
//             params.scoreThreshold.type, params.scoreThreshold.shape, params.scoreThreshold.data.data());
//         const auto nms = std::make_shared<opset4::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false,
//                                                                      params.expectedSelectedIndices.type);
//         const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
//         return f;
//     }
// };

// class ReferenceNMSRotated4TestWithoutConstants : public ReferenceNMSRotated4Test {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data, params.maxOutputBoxesPerClass.data,
//                      params.iouThreshold.data, params.scoreThreshold.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated4Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Parameter>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape);
//         const auto iou_threshold = std::make_shared<opset1::Parameter>(
//             params.iouThreshold.type, params.iouThreshold.shape);
//         const auto score_threshold = std::make_shared<opset1::Parameter>(
//             params.scoreThreshold.type, params.scoreThreshold.shape);
//         const auto nms = std::make_shared<opset4::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false,
//                                                                      params.expectedSelectedIndices.type);
//         const auto f = std::make_shared<Model>(nms->outputs(),
//                                                ParameterVector{boxes, scores, max_output_boxes_per_class,
//                                                                iou_threshold, score_threshold});
//         return f;
//     }
// };

// TEST_P(ReferenceNMSRotated4Test, CompareWithRefs) {
//     Exec();
// }

// TEST_P(ReferenceNMSRotated4TestWithoutConstants, CompareWithRefs) {
//     Exec();
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated4Params> generateParams4() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated4Params> params {
//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
//                 0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CENTER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_center_point_box_format"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
//                 0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_flipped_coordinates"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
//                 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{
//                 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_identical_boxes"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_limit_output_size"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_single_box"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {2, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {2, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0}))
//         .testcaseName(
//             "NMSRotated_two_batches"),

//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 2, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
//         .testcaseName(
//             "NMSRotated_two_classes"),
//     };
//     return params;
// }

// std::vector<NMSRotated4Params> generateCombinedParams4() {
//     const std::vector<std::vector<NMSRotated4Params>> generatedParams {
//         generateParams4<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams4<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams4<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated4Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated4Params> generateParams4WithoutConstants() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated4Params> params {
//         Builder4 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,  1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
//                 0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
//         .boxEncoding(
//             opset4::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores_without_constants"),
//     };
//     return params;
// }

// std::vector<NMSRotated4Params> generateCombinedParams4WithoutConstants() {
//     const std::vector<std::vector<NMSRotated4Params>> generatedParams {
//         generateParams4WithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4WithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4WithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams4WithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams4WithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams4WithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated4Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated4Test,
//     testing::ValuesIn(generateCombinedParams4()), ReferenceNMSRotated4Test::getTestCaseName);
// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated4TestWithoutConstants,
//     testing::ValuesIn(generateCombinedParams4WithoutConstants()), ReferenceNMSRotated4TestWithoutConstants::getTestCaseName);


// struct NMSRotated3Params {
//     reference_tests::Tensor boxes;
//     reference_tests::Tensor scores;
//     reference_tests::Tensor maxOutputBoxesPerClass;
//     reference_tests::Tensor iouThreshold;
//     reference_tests::Tensor scoreThreshold;
//     opset3::NMSRotated::BoxEncodingType boxEncoding;
//     reference_tests::Tensor expectedSelectedIndices;
//     std::string testcaseName;
// };

// struct Builder3 : ParamsBuilder<NMSRotated3Params> {
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, boxes);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, scores);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, maxOutputBoxesPerClass);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, iouThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, scoreThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, boxEncoding);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, expectedSelectedIndices);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder3, testcaseName);
// };

// class ReferenceNMSRotated3Test : public testing::TestWithParam<NMSRotated3Params>, public CommonReferenceTest {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

//     static std::string getTestCaseName(const testing::TestParamInfo<NMSRotated3Params>& obj) {
//         auto param = obj.param;
//         std::ostringstream result;
//         result << "bType=" << param.boxes.type;
//         result << "_bShape=" << param.boxes.shape;
//         result << "_sType=" << param.scores.type;
//         result << "_sShape=" << param.scores.shape;
//         result << "_esiType=" << param.expectedSelectedIndices.type;
//         result << "_esiShape=" << param.expectedSelectedIndices.shape;
//         if (param.testcaseName != "") {
//             result << "_=" << param.testcaseName;
//         }
//         return result.str();
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated3Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Constant>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape, params.maxOutputBoxesPerClass.data.data());
//         const auto iou_threshold = std::make_shared<opset1::Constant>(
//             params.iouThreshold.type, params.iouThreshold.shape, params.iouThreshold.data.data());
//         const auto score_threshold = std::make_shared<opset1::Constant>(
//             params.scoreThreshold.type, params.scoreThreshold.shape, params.scoreThreshold.data.data());
//         const auto nms = std::make_shared<opset3::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false,
//                                                                      params.expectedSelectedIndices.type);
//         const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
//         return f;
//     }
// };

// class ReferenceNMSRotated3TestWithoutConstants : public ReferenceNMSRotated3Test {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data, params.maxOutputBoxesPerClass.data,
//                      params.iouThreshold.data, params.scoreThreshold.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated3Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Parameter>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape);
//         const auto iou_threshold = std::make_shared<opset1::Parameter>(
//             params.iouThreshold.type, params.iouThreshold.shape);
//         const auto score_threshold = std::make_shared<opset1::Parameter>(
//             params.scoreThreshold.type, params.scoreThreshold.shape);
//         const auto nms = std::make_shared<opset3::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false,
//                                                                      params.expectedSelectedIndices.type);
//         const auto f = std::make_shared<Model>(nms->outputs(),
//                                                ParameterVector{boxes, scores, max_output_boxes_per_class,
//                                                                iou_threshold, score_threshold});
//         return f;
//     }
// };

// TEST_P(ReferenceNMSRotated3Test, CompareWithRefs) {
//     Exec();
// }

// TEST_P(ReferenceNMSRotated3TestWithoutConstants, CompareWithRefs) {
//     Exec();
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated3Params> generateParams3() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated3Params> params {
//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
//                 0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CENTER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_center_point_box_format"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
//                 0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_flipped_coordinates"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
//                 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{
//                 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_identical_boxes"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_limit_output_size"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_single_box"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores"),

//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 2, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
//         .testcaseName(
//             "NMSRotated_two_classes"),
//     };
//     return params;
// }

// std::vector<NMSRotated3Params> generateCombinedParams3() {
//     const std::vector<std::vector<NMSRotated3Params>> generatedParams {
//         generateParams3<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams3<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams3<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated3Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated3Params> generateParams3WithoutConstants() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated3Params> params {
//         Builder3 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,  1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
//                 0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
//         .boxEncoding(
//             opset3::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores_without_constants"),
//     };
//     return params;
// }

// std::vector<NMSRotated3Params> generateCombinedParams3WithoutConstants() {
//     const std::vector<std::vector<NMSRotated3Params>> generatedParams {
//         generateParams3WithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3WithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3WithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
//         generateParams3WithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams3WithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams3WithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated3Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated3Test,
//     testing::ValuesIn(generateCombinedParams3()), ReferenceNMSRotated3Test::getTestCaseName);
// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated3TestWithoutConstants,
//     testing::ValuesIn(generateCombinedParams3WithoutConstants()), ReferenceNMSRotated3TestWithoutConstants::getTestCaseName);


// struct NMSRotated1Params {
//     reference_tests::Tensor boxes;
//     reference_tests::Tensor scores;
//     reference_tests::Tensor maxOutputBoxesPerClass;
//     reference_tests::Tensor iouThreshold;
//     reference_tests::Tensor scoreThreshold;
//     opset1::NMSRotated::BoxEncodingType boxEncoding;
//     reference_tests::Tensor expectedSelectedIndices;
//     std::string testcaseName;
// };

// struct Builder1 : ParamsBuilder<NMSRotated1Params> {
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, boxes);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, scores);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, maxOutputBoxesPerClass);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, iouThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, scoreThreshold);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, boxEncoding);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, expectedSelectedIndices);
//     REFERENCE_TESTS_ADD_SET_PARAM(Builder1, testcaseName);
// };

// class ReferenceNMSRotated1Test : public testing::TestWithParam<NMSRotated1Params>, public CommonReferenceTest {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

//     static std::string getTestCaseName(const testing::TestParamInfo<NMSRotated1Params>& obj) {
//         auto param = obj.param;
//         std::ostringstream result;
//         result << "bType=" << param.boxes.type;
//         result << "_bShape=" << param.boxes.shape;
//         result << "_sType=" << param.scores.type;
//         result << "_sShape=" << param.scores.shape;
//         result << "_esiType=" << param.expectedSelectedIndices.type;
//         result << "_esiShape=" << param.expectedSelectedIndices.shape;
//         if (param.testcaseName != "") {
//             result << "_=" << param.testcaseName;
//         }
//         return result.str();
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated1Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Constant>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape, params.maxOutputBoxesPerClass.data.data());
//         const auto iou_threshold = std::make_shared<opset1::Constant>(
//             params.iouThreshold.type, params.iouThreshold.shape, params.iouThreshold.data.data());
//         const auto score_threshold = std::make_shared<opset1::Constant>(
//             params.scoreThreshold.type, params.scoreThreshold.shape, params.scoreThreshold.data.data());
//         const auto nms = std::make_shared<opset1::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false);
//         const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
//         return f;
//     }
// };

// class ReferenceNMSRotated1TestWithoutConstants : public ReferenceNMSRotated1Test {
// public:
//     void SetUp() override {
//         auto params = GetParam();
//         function = CreateFunction(params);
//         inputData = {params.boxes.data, params.scores.data, params.maxOutputBoxesPerClass.data,
//                      params.iouThreshold.data, params.scoreThreshold.data};
//         refOutData = {params.expectedSelectedIndices.data};
//     }

// private:
//     static std::shared_ptr<Model> CreateFunction(const NMSRotated1Params& params) {
//         const auto boxes = std::make_shared<opset1::Parameter>(params.boxes.type, params.boxes.shape);
//         const auto scores = std::make_shared<opset1::Parameter>(params.scores.type, params.scores.shape);
//         const auto max_output_boxes_per_class = std::make_shared<opset1::Parameter>(
//             params.maxOutputBoxesPerClass.type, params.maxOutputBoxesPerClass.shape);
//         const auto iou_threshold = std::make_shared<opset1::Parameter>(
//             params.iouThreshold.type, params.iouThreshold.shape);
//         const auto score_threshold = std::make_shared<opset1::Parameter>(
//             params.scoreThreshold.type, params.scoreThreshold.shape);
//         const auto nms = std::make_shared<opset1::NMSRotated>(boxes,
//                                                                      scores,
//                                                                      max_output_boxes_per_class,
//                                                                      iou_threshold,
//                                                                      score_threshold,
//                                                                      params.boxEncoding,
//                                                                      false);
//         const auto f = std::make_shared<Model>(nms->outputs(),
//                                                ParameterVector{boxes, scores, max_output_boxes_per_class,
//                                                                iou_threshold, score_threshold});
//         return f;
//     }
// };

// TEST_P(ReferenceNMSRotated1Test, CompareWithRefs) {
//     Exec();
// }

// TEST_P(ReferenceNMSRotated1TestWithoutConstants, CompareWithRefs) {
//     Exec();
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated1Params> generateParams1() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated1Params> params {
//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
//                 0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CENTER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_center_point_box_format"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
//                 0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_flipped_coordinates"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
//                 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
//                 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{
//                 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_identical_boxes"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_limit_output_size"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{0.9}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_single_box"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 0, 5}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores"),

//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
//                 0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 2, 6}, std::vector<T>{
//                 0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {4, 3}, std::vector<T_IND>{
//                 0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0}))
//         .testcaseName(
//             "NMSRotated_two_classes"),
//     };
//     return params;
// }

// std::vector<NMSRotated1Params> generateCombinedParams1() {
//     const std::vector<std::vector<NMSRotated1Params>> generatedParams {
//         generateParams1<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams1<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams1<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated1Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
// std::vector<NMSRotated1Params> generateParams1WithoutConstants() {
//     using T = typename element_type_traits<ET>::value_type;
//     using T_BOX = typename element_type_traits<ET_BOX>::value_type;
//     using T_TH = typename element_type_traits<ET_TH>::value_type;
//     using T_IND = typename element_type_traits<ET_IND>::value_type;
//     std::vector<NMSRotated1Params> params {
//         Builder1 {}
//         .boxes(
//             reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{
//                 0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 0.1f,  1.0f, 1.1f,  0.0f, -0.1f,  1.0f, 0.9f,
//                 0.0f, 10.0f, 1.0f, 11.0f, 0.0f, 10.1f, 1.0f, 11.1f, 0.0f, 100.0f, 1.0f, 101.0f}))
//         .scores(
//             reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{
//                 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f}))
//         .maxOutputBoxesPerClass(
//             reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{1}))
//         .iouThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
//         .scoreThreshold(
//             reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.2f}))
//         .boxEncoding(
//             opset1::NMSRotated::BoxEncodingType::CORNER)
//         .expectedSelectedIndices(
//             reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 3}))
//         .testcaseName(
//             "NMSRotated_suppress_by_IOU_and_scores_without_constants"),
//     };
//     return params;
// }

// std::vector<NMSRotated1Params> generateCombinedParams1WithoutConstants() {
//     const std::vector<std::vector<NMSRotated1Params>> generatedParams {
//         generateParams1WithoutConstants<element::Type_t::bf16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams1WithoutConstants<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//         generateParams1WithoutConstants<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
//     };
//     std::vector<NMSRotated1Params> combinedParams;

//     for (const auto& params : generatedParams) {
//         combinedParams.insert(combinedParams.end(), params.begin(), params.end());
//     }
//     return combinedParams;
// }

// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated1Test,
//     testing::ValuesIn(generateCombinedParams1()), ReferenceNMSRotated1Test::getTestCaseName);
// INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs, ReferenceNMSRotated1TestWithoutConstants,
//     testing::ValuesIn(generateCombinedParams1WithoutConstants()), ReferenceNMSRotated1TestWithoutConstants::getTestCaseName);



} // namespace
