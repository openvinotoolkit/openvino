// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multiclass_nms.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct MulticlassNmsParams {
    MulticlassNmsParams(const int nms_top_k,
                        const float iou_threshold,
                        const float score_threshold,
                        const op::v8::MulticlassNms::SortResultType sort_result_type,
                        const int keep_top_k,
                        const int background_class,
                        const float nms_eta,
                        const ov::element::Type output_type,
                        const bool sort_result_across_batch,
                        const bool normalized,
                        const reference_tests::Tensor& boxes,
                        const reference_tests::Tensor& scores,
                        const reference_tests::Tensor& expectedSelectedScores,
                        const reference_tests::Tensor& expectedSelectedIndices,
                        const reference_tests::Tensor& expectedValidOutputs,
                        const std::string& testcaseName = "")
        : nms_top_k(nms_top_k),
          iou_threshold(iou_threshold),
          score_threshold(score_threshold),
          sort_result_type(sort_result_type),
          keep_top_k(keep_top_k),
          background_class(background_class),
          nms_eta(nms_eta),
          output_type(output_type),
          sort_result_across_batch(sort_result_across_batch),
          normalized(normalized),
          boxes(boxes),
          scores(scores),
          expectedSelectedScores(expectedSelectedScores),
          expectedSelectedIndices(expectedSelectedIndices),
          expectedValidOutputs(expectedValidOutputs),
          testcaseName(testcaseName) {}

    int nms_top_k;
    float iou_threshold;
    float score_threshold;
    op::v8::MulticlassNms::SortResultType sort_result_type;
    int keep_top_k;
    int background_class;
    float nms_eta;
    ov::element::Type output_type;

    bool sort_result_across_batch = false;
    bool normalized = true;

    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor expectedSelectedScores;
    reference_tests::Tensor expectedSelectedIndices;
    reference_tests::Tensor expectedValidOutputs;
    std::string testcaseName;
};

class ReferenceMulticlassNmsTest : public testing::TestWithParam<MulticlassNmsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedScores.data,
                      params.expectedSelectedIndices.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj) {
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
    static std::shared_ptr<Model> CreateFunction(const MulticlassNmsParams& params) {
        op::v8::MulticlassNms::Attributes attrs;
        attrs.nms_top_k = params.nms_top_k;
        attrs.iou_threshold = params.iou_threshold;
        attrs.score_threshold = params.score_threshold;
        attrs.sort_result_type = params.sort_result_type;
        attrs.keep_top_k = params.keep_top_k;
        attrs.background_class = params.background_class;
        attrs.nms_eta = params.nms_eta;
        attrs.output_type = params.output_type;
        attrs.sort_result_across_batch = params.sort_result_across_batch;
        attrs.normalized = params.normalized;
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, PartialShape::dynamic());
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, PartialShape::dynamic());
        const auto nms = std::make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

TEST_P(ReferenceMulticlassNmsTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<MulticlassNmsParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<MulticlassNmsParams> params{
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {4, 6}, std::vector<T_TH>{0.00,  0.95, 0.00, 10.00, 1.00, 11.00, 1.00,
                                                                     0.95,  0.00, 0.00, 1.00,  1.00, 0.00,  0.90,
                                                                     0.00,  0.00, 1.00, 1.00,  1.00, 0.80,  0.00,
                                                                     10.00, 1.00, 11.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {4, 1}, std::vector<T_IND>{3, 0, 0, 3}),        // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}),                    // expected_valid_outputs
            "multiclass_nms_by_score"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            -1,                                              // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {4, 6}, std::vector<T_TH>{0.00,  0.95, 0.00, 10.00, 1.00, 11.00, 0.00,
                                                                     0.90,  0.00, 0.00, 1.00,  1.00, 1.00,  0.95,
                                                                     0.00,  0.00, 1.00, 1.00,  1.00, 0.80,  0.00,
                                                                     10.00, 1.00, 11.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {4, 1}, std::vector<T_IND>{3, 0, 0, 3}),        // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}),                    // expected_valid_outputs
            "multiclass_nms_by_class_id"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            -1,                                              // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {4, 6}, std::vector<T_TH>{0.00,  0.95, 0.00, 10.00, 1.00, 11.00, 0.00,
                                                                     0.90,  0.00, 0.00, 1.00,  1.00, 1.00,  0.95,
                                                                     0.00,  0.00, 1.00, 1.00,  1.00, 0.80,  0.00,
                                                                     10.00, 1.00, 11.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {4, 1}, std::vector<T_IND>{3, 0, 0, 3}),        // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{4}),                    // expected_valid_outputs
            "multiclass_nms_output_type_i32"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {8, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                                  0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                  0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                                  0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND,
                                    {8, 1},
                                    std::vector<T_IND>{3, 0, 0, 3, 9, 6, 6, 9}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),       // expected_valid_outputs
            "multiclass_nms_two_batches_two_classes_by_score"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            -1,                                              // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {8, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                                  1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                  0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                                  1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND,
                                    {8, 1},
                                    std::vector<T_IND>{3, 0, 0, 3, 9, 6, 6, 9}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),       // expected_valid_outputs
            "multiclass_nms_two_batches_two_classes_by_class_id"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            true,                                          // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {8, 6},
                std::vector<T_TH>{
                    0.00,  0.95, 0.00,  10.00, 1.00,  11.00, 1.00,  0.95, 0.00, 0.00, 1.00, 1.00, 0.00,
                    0.95,  0.00, 10.00, 1.00,  11.00, 1.00,  0.95,  0.00, 0.00, 1.00, 1.00, 0.00, 0.90,
                    0.00,  0.00, 1.00,  1.00,  0.00,  0.90,  0.00,  0.00, 1.00, 1.00, 1.00, 0.80, 0.00,
                    10.00, 1.00, 11.00, 1.00,  0.80,  0.00,  10.00, 1.00, 11.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND,
                                    {8, 1},
                                    std::vector<T_IND>{3, 0, 9, 6, 0, 6, 3, 9}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),       // expected_valid_outputs
            "multiclass_nms_two_batches_two_classes_by_score_cross_batch"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            -1,                                              // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            true,                                            // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {8, 6},
                std::vector<T_TH>{
                    0.00, 0.95, 0.00,  10.00, 1.00,  11.00, 0.00,  0.90,  0.00, 0.00,  1.00, 1.00, 0.00,
                    0.95, 0.00, 10.00, 1.00,  11.00, 0.00,  0.90,  0.00,  0.00, 1.00,  1.00, 1.00, 0.95,
                    0.00, 0.00, 1.00,  1.00,  1.00,  0.80,  0.00,  10.00, 1.00, 11.00, 1.00, 0.95, 0.00,
                    0.00, 1.00, 1.00,  1.00,  0.80,  0.00,  10.00, 1.00,  11.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND,
                                    {8, 1},
                                    std::vector<T_IND>{3, 0, 9, 6, 0, 3, 6, 9}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),       // expected_valid_outputs
            "multiclass_nms_two_batches_two_classes_by_class_id_cross_batch"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0,  0.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  0.9, 1.0,  -0.1, 0.0,   10.0, 1.0,  11.0, 1.0, 10.1,
                                                                  0.0, 11.1, 1.0,  101.0, 0.0,  100.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {3, 6},
                                    std::vector<T_TH>{0.00,
                                                      0.95,
                                                      0.00,
                                                      10.00,
                                                      1.00,
                                                      11.00,
                                                      0.00,
                                                      0.90,
                                                      1.00,
                                                      1.00,
                                                      0.00,
                                                      0.00,
                                                      0.00,
                                                      0.75,
                                                      0.00,
                                                      0.10,
                                                      1.00,
                                                      1.10}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {3, 1}, std::vector<T_IND>{3, 0, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}),           // expected_valid_outputs
            "multiclass_nms_flipped_coordinates"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(
                ET,
                {1, 10, 4},
                std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                               1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                               0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}),  // boxes
            reference_tests::Tensor(ET_TH,
                                    {1, 1, 10},
                                    std::vector<T_TH>{0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}),  // scores
            reference_tests::Tensor(ET_TH, {1, 6}, std::vector<T_TH>{0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}),     // expected_valid_outputs
            "multiclass_nms_identical_boxes"),
        MulticlassNmsParams(
            2,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {2, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {2, 1}, std::vector<T_IND>{3, 0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}),        // expected_valid_outputs
            "multiclass_nms_limit_output_size"),
        MulticlassNmsParams(
            3,                                                                           // nms_top_k
            0.5f,                                                                        // iou_threshold
            0.0f,                                                                        // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,                                // sort_result_type
            -1,                                                                          // keep_top_k
            -1,                                                                          // background_class
            1.0f,                                                                        // nms_eta
            ET_IND,                                                                      // output_type
            false,                                                                       // sort_result_across_batch
            true,                                                                        // normalized
            reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}),  // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 1}, std::vector<T_TH>{0.9}),           // scores
            reference_tests::Tensor(ET_TH,
                                    {1, 6},
                                    std::vector<T_TH>{0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}),     // expected_valid_outputs
            "multiclass_nms_single_box"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.2f,                                          // iou_threshold
            0.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {2, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {2, 1}, std::vector<T_IND>{3, 0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}),        // expected_valid_outputs
            "multiclass_nms_by_IOU"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            0.95f,                                         // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.96, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {1, 6},
                std::vector<T_TH>{0.00, 0.96, 0.00, 10.00, 1.00, 11.00}),    // expected_selected_scores
            reference_tests::Tensor(ET_IND, {1, 1}, std::vector<T_IND>{3}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}),     // expected_valid_outputs
            "multiclass_nms_by_IOU_and_scores"),
        MulticlassNmsParams(
            3,                                             // nms_top_k
            0.5f,                                          // iou_threshold
            2.0f,                                          // score_threshold
            op::v8::MulticlassNms::SortResultType::SCORE,  // sort_result_type
            -1,                                            // keep_top_k
            -1,                                            // background_class
            1.0f,                                          // nms_eta
            ET_IND,                                        // output_type
            false,                                         // sort_result_across_batch
            true,                                          // normalized
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {0, 6}, std::vector<T_TH>{}),    // expected_selected_scores
            reference_tests::Tensor(ET_IND, {0, 1}, std::vector<T_IND>{}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{0}),    // expected_valid_outputs
            "multiclass_nms_no_output"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            0,                                               // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {4, 6}, std::vector<T_TH>{1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                                                                     1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                                                     1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                                                                     1.00, 0.80, 0.00, 10.00, 1.00, 11.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {4, 1}, std::vector<T_IND>{0, 3, 6, 9}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{2, 2}),           // expected_valid_outputs
            "multiclass_nms_by_background"),
        MulticlassNmsParams(
            3,                                               // nms_top_k
            0.5f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            3,                                               // keep_top_k
            -1,                                              // background_class
            1.0f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {6, 6},
                std::vector<T_TH>{
                    0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00,  0.00, 1.00,  1.00, 1.00,
                    0.95, 0.00, 0.00, 1.00,  1.00, 0.00,  0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90,
                    0.00, 0.00, 1.00, 1.00,  1.00, 0.95,  0.00, 0.00, 1.00,  1.00}),        // expected_selected_scores
            reference_tests::Tensor(ET_IND, {6, 1}, std::vector<T_IND>{3, 0, 0, 9, 6, 6}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{3, 3}),                 // expected_valid_outputs
            "multiclass_nms_by_keep_top_k"),
        MulticlassNmsParams(
            -1,                                              // nms_top_k
            1.0f,                                            // iou_threshold
            0.0f,                                            // score_threshold
            op::v8::MulticlassNms::SortResultType::CLASSID,  // sort_result_type
            -1,                                              // keep_top_k
            -1,                                              // background_class
            0.1f,                                            // nms_eta
            ET_IND,                                          // output_type
            false,                                           // sort_result_across_batch
            true,                                            // normalized
            reference_tests::Tensor(
                ET,
                {2, 6, 4},
                std::vector<T>{0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                               0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                               0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
            reference_tests::Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                                                                        0.9,  0.75, 0.6, 0.95, 0.5, 0.3,
                                                                        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {12, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00,  1.00, 11.00,  0.00, 0.90, 0.00, 0.00,   1.00, 1.00,
                                  0.00, 0.30, 0.00, 100.00, 1.00, 101.00, 1.00, 0.95, 0.00, 0.00,   1.00, 1.00,
                                  1.00, 0.80, 0.00, 10.00,  1.00, 11.00,  1.00, 0.30, 0.00, 100.00, 1.00, 101.00,
                                  0.00, 0.95, 0.00, 10.00,  1.00, 11.00,  0.00, 0.90, 0.00, 0.00,   1.00, 1.00,
                                  0.00, 0.30, 0.00, 100.00, 1.00, 101.00, 1.00, 0.95, 0.00, 0.00,   1.00, 1.00,
                                  1.00, 0.80, 0.00, 10.00,  1.00, 11.00,  1.00, 0.30, 0.00, 100.00, 1.00, 101.00}),
            // expected_selected_scores
            reference_tests::Tensor(
                ET_IND,
                {12, 1},
                std::vector<T_IND>{3, 0, 5, 0, 3, 5, 9, 6, 11, 6, 9, 11}),   // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{6, 6}),  // expected_valid_outputs
            "multiclass_nms_by_nms_eta"),
    };
    return params;
}

std::vector<MulticlassNmsParams> generateCombinedParams() {
    const std::vector<std::vector<MulticlassNmsParams>> generatedParams{
        generateParams<element::Type_t::bf16, element::Type_t::bf16, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::f16, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::bf16, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::f16, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<MulticlassNmsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_MulticlassNms_With_Hardcoded_Refs,
                         ReferenceMulticlassNmsTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceMulticlassNmsTest::getTestCaseName);
}  // namespace
