// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

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
    bool sortResultsDescending = true;
    bool clockwise = true;
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
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, sortResultsDescending);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, clockwise);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedSelectedIndices);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedSelectedScores);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedValidOutputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceNMSRotatedTest : public testing::TestWithParam<NMSRotatedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateModel(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NMSRotatedParams>& obj) {
        const auto& param = obj.param;
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
    static std::shared_ptr<Model> CreateModel(const NMSRotatedParams& params) {
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
        const auto nms = std::make_shared<op::v13::NMSRotated>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               params.sortResultsDescending,
                                                               params.expectedSelectedIndices.type,
                                                               params.clockwise);
        return std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
    }
};

class ReferenceNMSRotatedTestWithoutConstants : public ReferenceNMSRotatedTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateModel(params);
        inputData = {params.boxes.data,
                     params.scores.data,
                     params.maxOutputBoxesPerClass.data,
                     params.iouThreshold.data,
                     params.scoreThreshold.data};
        refOutData = {params.expectedSelectedIndices.data,
                      params.expectedSelectedScores.data,
                      params.expectedValidOutputs.data};
    }

private:
    static std::shared_ptr<Model> CreateModel(const NMSRotatedParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, params.boxes.shape);
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, params.scores.shape);
        const auto max_output_boxes_per_class =
            std::make_shared<op::v0::Parameter>(params.maxOutputBoxesPerClass.type,
                                                params.maxOutputBoxesPerClass.shape);
        const auto iou_threshold =
            std::make_shared<op::v0::Parameter>(params.iouThreshold.type, params.iouThreshold.shape);
        const auto score_threshold =
            std::make_shared<op::v0::Parameter>(params.scoreThreshold.type, params.scoreThreshold.shape);
        const auto nms = std::make_shared<op::v13::NMSRotated>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               params.sortResultsDescending,
                                                               params.expectedSelectedIndices.type,
                                                               params.clockwise);
        return std::make_shared<Model>(
            nms->outputs(),
            ParameterVector{boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold});
    }
};

TEST_P(ReferenceNMSRotatedTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceNMSRotatedTestWithoutConstants, CompareWithRefs) {
    Exec();
}

// clang-format off
// To make the test data shape more readable
template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NMSRotatedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NMSRotatedParams> params{
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 4, 5}, std::vector<T>{/*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {3, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1, 0, 0, 0}))  // batch 0, class 0, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH,
                                        {3, 3},
                                        std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7, 0.0, 0.0, 0.65}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("NMSRotated_new_rotation_basic"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 4, 5}, std::vector<T>{/*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {1, 2, 4}, std::vector<T>{/*0*/ 0.65, 0.7, 0.55, 0.96, /*1*/ 0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .sortResultsDescending(false)
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {6, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1, 3, 0, 1, 1, 0, 1, 0}))  // batch, class, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH,
                                        {6, 3},
                                        std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7, 0.0, 0.0, 0.65, 0.0, 1.0, 0.96, 0.0, 1.0, 0.7, 0.0, 1.0, 0.65}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{6}))
            .testcaseName("NMSRotated_new_rotation_class_2"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {2, 4, 5}, std::vector<T>{/* First batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6,
                                                                         /* Second batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {2, 1, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .sortResultsDescending(false)
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {6, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1, 0, 0, 0,
                                   1, 0, 3, 1, 0, 1, 1, 0, 0}))  // batch, class, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH,
                                        {6, 3},
                                        std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7, 0.0, 0.0, 0.65, 1.0, 0.0, 0.96, 1.0, 0.0, 0.7, 1.0, 0.0, 0.65}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{6}))
            .testcaseName("NMSRotated_new_rotation_batch_2"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {2, 4, 5}, std::vector<T>{/* First batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6,
                                                                         /* Second batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {2, 2, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .sortResultsDescending(false)
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {12, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1, 0, 0, 0,
                                   0, 1, 3, 0, 1, 1, 0, 1, 0,
                                   1, 0, 3, 1, 0, 1, 1, 0, 0,
                                   1, 1, 3, 1, 1, 1, 1, 1, 0}))  // batch, class, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH,
                                        {12, 3},
                                        std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7, 0.0, 0.0, 0.65,
                                                          0.0, 1.0, 0.96, 0.0, 1.0, 0.7, 0.0, 1.0, 0.65,
                                                          1.0, 0.0, 0.96, 1.0, 0.0, 0.7, 1.0, 0.0, 0.65,
                                                          1.0, 1.0, 0.96, 1.0, 1.0, 0.7, 1.0, 1.0, 0.65,
                                                          }))

            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{12}))
            .testcaseName("NMSRotated_new_rotation_batch_2_class_2_sort_attr_false"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {2, 4, 5}, std::vector<T>{/* First batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6,
                                                                         /* Second batch */
                                                                         /*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {2, 2, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96,
                                                                          0.65, 0.7, 0.55, 0.96, 0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .sortResultsDescending(true)

            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {12, 3},
                std::vector<T_IND>{0, 0, 3, 0, 1, 3, 1, 0, 3, 1, 1, 3,
                                   0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                                   0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0}))  // batch, class, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH,
                                        {12, 3},
                                        std::vector<T_TH>{0.0, 0.0, 0.96,
                                                          0.0, 1.0, 0.96,
                                                          1.0, 0.0, 0.96,
                                                          1.0, 1.0, 0.96,

                                                          0.0, 0.0, 0.7,
                                                          0.0, 1.0, 0.7,
                                                          1.0, 0.0, 0.7,
                                                          1.0, 1.0, 0.7,

                                                          0.0, 0.0, 0.65,
                                                          0.0, 1.0, 0.65,
                                                          1.0, 0.0, 0.65,
                                                          1.0, 1.0, 0.65
                                                          }))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{12}))
            .testcaseName("NMSRotated_new_rotation_batch_2_class_2_sort_attr_true"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 4, 5}, std::vector<T>{/*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{2}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {2, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1}))  // batch 0, class 0, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_basic_max_out_2"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 4, 5}, std::vector<T>{/*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.67f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {2, 3},
                std::vector<T_IND>{0, 0, 3, 0, 0, 1}))  // batch 0, class 0, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.7}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_basic_score_tresh"),
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 4, 5}, std::vector<T>{/*0*/ 7.0, 4.0, 8.0,  7.0,  0.5,
                                                                         /*1*/ 4.0, 7.0, 9.0,  11.0, 0.6,
                                                                         /*2*/ 4.0, 8.0, 10.0, 12.0, 0.3,
                                                                         /*3*/ 2.0, 5.0, 13.0, 7.0,  0.6}))
            .scores(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.65, 0.7, 0.55, 0.96}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.3f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.96, 0.0, 0.0, 0.65}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_2"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 8.0, 11.5, 4.0, 3.0, 0.5236, /*1*/ 11.0, 15.0, 8.0, 2.0, 0.7854}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.8}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 0, 0, 0, 1}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.8, 0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_3"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 8.0, 11.5, 4.0, 3.0, 0.5236, /*1*/ 11.0, 15.0, 8.0, 2.0, 0.7854}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.8}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_4"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 8.0, 11.5, 4.0, 3.0, 0.5236, /*1*/ 11.0, 15.0, 8.0, 2.0, 0.7854}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.7, 0.8}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 1}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_5"),
        Builder{}
            .boxes(
                reference_tests::Tensor(ET,
                                        {1, 2, 5},
                                        std::vector<T>{/*0*/ 23.0, 3.5, 4.0, 5.0, 2.9, /*1*/ 22.0, 3.5, 4.0, 3.0, 5.3}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.7, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {1, 3},
                std::vector<T_IND>{0, 0, 1}))  // batch 0, class 0, box_id (sorted max score first)
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_6"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 6.0, 34.0, 4.0, 8.0, -0.7854, /*1*/ 9.0, 32, 2.0, 4.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.7}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .clockwise(true)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 0, 0, 0, 1}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.8, 0.0, 0.0, 0.7}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_negative_cw"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 6.0, 34.0, 4.0, 8.0, -0.7854, /*1*/ 9.0, 32, 2.0, 4.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.7}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .clockwise(false)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_negative_ccw"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 9.0, 32, 2.0, 4.0, 0.0, /*1*/ 6.0, 34.0, 4.0, 8.0, -0.7854}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.7}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .clockwise(false)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_negative_ccw_reorder"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 6.0, 34.0, 4.0, 8.0, 0.7854, /*1*/ 9.0, 32, 2.0, 4.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.7}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .clockwise(false)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {2, 3}, std::vector<T_IND>{0, 0, 0, 0, 0, 1}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {2, 3}, std::vector<T_TH>{0.0, 0.0, 0.8, 0.0, 0.0, 0.7}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}))
            .testcaseName("NMSRotated_new_rotation_positive_ccw"),
        Builder{}
            .boxes(reference_tests::Tensor(
                ET,
                {1, 2, 5},
                std::vector<T>{/*0*/ 6.0, 34.0, 4.0, 8.0, 0.7854, /*1*/ 9.0, 32, 2.0, 4.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.8, 0.7}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.1f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .clockwise(true)
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 0}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.8}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_positive_cw"),
        Builder{}
            .boxes(
                reference_tests::Tensor(ET,
                                        {1, 2, 5},
                                        std::vector<T>{/*0*/ 23.0, 3.5, 4.0, 5.0, 2.9, /*1*/ 22.0, 3.5, 4.0, 3.0, 5.3}))
            .scores(reference_tests::Tensor(ET, {1, 1, 2}, std::vector<T>{0.7, 0.9}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(ET_IND, {1, 3}, std::vector<T_IND>{0, 0, 1}))
            .expectedSelectedScores(reference_tests::Tensor(ET_TH, {1, 3}, std::vector<T_TH>{0.0, 0.0, 0.9}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}))
            .testcaseName("NMSRotated_new_rotation_7"),
        Builder{}
            .boxes(reference_tests::Tensor(ET,
                                           {1, 4, 5},
                                           std::vector<T>{
                                               /*0*/ 23.0, 3.5, 4.0, 5.0, 2.9, /*1*/ 11.0, 15.0, 8.0, 2.0, 0.7854,
                                               /*2*/ 22.0, 3.5, 4.0, 3.0, 5.3, /*3*/ 8.0,  11.5, 4.0, 3.0, 0.5236,
                                           }))
            .scores(reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.9, 0.7, 0.6, 0.8}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{5000}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.4f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(reference_tests::Tensor(
                ET_IND,
                {3, 3},
                std::vector<T_IND>{0, 0, 0, 0, 0, 3, 0, 0, 1}))  // batch 0, class 0, box_id (sorted max score first)
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.9, 0.0, 0.0, 0.8, 0.0, 0.0, 0.7}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("NMSRotated_new_rotation_8"),

        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 5}, std::vector<T>{/*0*/ 0.5, 0.5,   1.0, 1.0, 0.0,
                                                                         /*1*/ 0.5, 0.6,   1.0, 1.0, 0.0,
                                                                         /*2*/ 0.5, 0.4,   1.0, 1.0, 0.0,
                                                                         /*3*/ 0.5, 10.5,  1.0, 1.0, 0.0,
                                                                         /*4*/ 0.5, 10.6,  1.0, 1.0, 0.0,
                                                                         /*5*/ 0.5, 100.5, 1.0, 1.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("NMSRotated_center_point_zero_angle"),

    };
    return params;
}
// clang-format on

std::vector<NMSRotatedParams> generateCombinedParams() {
    const std::vector<std::vector<NMSRotatedParams>> generatedParams{
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::i32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<NMSRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        std::move(params.begin(), params.end(), std::back_inserter(combinedParams));
    }
    return combinedParams;
}

template <element::Type_t ET, element::Type_t ET_BOX, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<NMSRotatedParams> generateParamsWithoutConstants() {
    using T = typename element_type_traits<ET>::value_type;
    using T_BOX = typename element_type_traits<ET_BOX>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<NMSRotatedParams> params{
        Builder{}
            .boxes(reference_tests::Tensor(ET, {1, 6, 5}, std::vector<T>{/*0*/ 0.5, 0.5,   1.0, 1.0, 0.0,
                                                                         /*1*/ 0.5, 0.6,   1.0, 1.0, 0.0,
                                                                         /*2*/ 0.5, 0.4,   1.0, 1.0, 0.0,
                                                                         /*3*/ 0.5, 10.5,  1.0, 1.0, 0.0,
                                                                         /*4*/ 0.5, 10.6,  1.0, 1.0, 0.0,
                                                                         /*5*/ 0.5, 100.5, 1.0, 1.0, 0.0}))
            .scores(reference_tests::Tensor(ET, {1, 1, 6}, std::vector<T>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}))
            .maxOutputBoxesPerClass(reference_tests::Tensor(ET_BOX, {}, std::vector<T_BOX>{3}))
            .iouThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.5f}))
            .scoreThreshold(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .softNmsSigma(reference_tests::Tensor(ET_TH, {}, std::vector<T_TH>{0.0f}))
            .expectedSelectedIndices(
                reference_tests::Tensor(ET_IND, {3, 3}, std::vector<T_IND>{0, 0, 3, 0, 0, 0, 0, 0, 5}))
            .expectedSelectedScores(
                reference_tests::Tensor(ET_TH, {3, 3}, std::vector<T_TH>{0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3}))
            .expectedValidOutputs(reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}))
            .testcaseName("NMSRotated_suppress_by_IOU_and_scores_without_constants"),
    };
    return params;
}

std::vector<NMSRotatedParams> generateCombinedParamsWithoutConstants() {
    const std::vector<std::vector<NMSRotatedParams>> generatedParams{
        generateParamsWithoutConstants<element::Type_t::f32,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i32>(),
        generateParamsWithoutConstants<element::Type_t::f16,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i64>(),
        generateParamsWithoutConstants<element::Type_t::f32,
                                       element::Type_t::i32,
                                       element::Type_t::f32,
                                       element::Type_t::i64>(),
    };
    std::vector<NMSRotatedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs,
                         ReferenceNMSRotatedTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceNMSRotatedTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_NMSRotated_With_Hardcoded_Refs,
                         ReferenceNMSRotatedTestWithoutConstants,
                         testing::ValuesIn(generateCombinedParamsWithoutConstants()),
                         ReferenceNMSRotatedTestWithoutConstants::getTestCaseName);

}  // namespace
