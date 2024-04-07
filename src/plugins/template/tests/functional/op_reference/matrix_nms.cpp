// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct MatrixNmsParams {
    MatrixNmsParams(const op::v8::MatrixNms::Attributes& attrs,
                    const reference_tests::Tensor& boxes,
                    const reference_tests::Tensor& scores,
                    const reference_tests::Tensor& expectedSelectedScores,
                    const reference_tests::Tensor& expectedSelectedIndices,
                    const reference_tests::Tensor& expectedValidOutputs,
                    const std::string& testcaseName = "")
        : attrs(attrs),
          boxes(boxes),
          scores(scores),
          expectedSelectedScores(expectedSelectedScores),
          expectedSelectedIndices(expectedSelectedIndices),
          expectedValidOutputs(expectedValidOutputs),
          testcaseName(testcaseName) {}

    op::v8::MatrixNms::Attributes attrs;
    reference_tests::Tensor boxes;
    reference_tests::Tensor scores;
    reference_tests::Tensor expectedSelectedScores;
    reference_tests::Tensor expectedSelectedIndices;
    reference_tests::Tensor expectedValidOutputs;
    std::string testcaseName;
};

class ReferenceMatrixNmsTest : public testing::TestWithParam<MatrixNmsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.boxes.data, params.scores.data};
        refOutData = {params.expectedSelectedScores.data,
                      params.expectedSelectedIndices.data,
                      params.expectedValidOutputs.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<MatrixNmsParams>& obj) {
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
    static std::shared_ptr<Model> CreateFunction(const MatrixNmsParams& params) {
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, PartialShape::dynamic());
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, PartialShape::dynamic());
        const auto nms = std::make_shared<op::v8::MatrixNms>(boxes, scores, params.attrs);
        const auto f = std::make_shared<Model>(nms->outputs(), ParameterVector{boxes, scores});
        return f;
    }
};

TEST_P(ReferenceMatrixNmsTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_TH, element::Type_t ET_IND>
std::vector<MatrixNmsParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_TH = typename element_type_traits<ET_TH>::value_type;
    using T_IND = typename element_type_traits<ET_IND>::value_type;
    std::vector<MatrixNmsParams> params{
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                0,                                         // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {3, 6},
                                    std::vector<T_TH>{1.00,
                                                      0.95,
                                                      0.00,
                                                      0.00,
                                                      1.00,
                                                      1.00,
                                                      1.00,
                                                      0.8,
                                                      0.00,
                                                      10.00,
                                                      1.00,
                                                      11.00,
                                                      1.00,
                                                      0.13636364,
                                                      0.0,
                                                      0.1,
                                                      1.0,
                                                      1.1}),                       // expected_selected_scores
            reference_tests::Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}),           // expected_valid_outputs
            "matrix_nms_output_type_i64"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                0,                                         // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {3, 6},
                                    std::vector<T_TH>{1.00,
                                                      0.95,
                                                      0.00,
                                                      0.00,
                                                      1.00,
                                                      1.00,
                                                      1.00,
                                                      0.8,
                                                      0.00,
                                                      10.00,
                                                      1.00,
                                                      11.00,
                                                      1.00,
                                                      0.13636364,
                                                      0.0,
                                                      0.1,
                                                      1.0,
                                                      1.1}),                       // expected_selected_scores
            reference_tests::Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}),           // expected_valid_outputs
            "matrix_nms_output_type_i32"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,    // sort_result_type
                false,                                       // sort_result_across_batch
                ET_IND,                                      // output_type
                0.0f,                                        // score_threshold
                3,                                           // nms_top_k
                -1,                                          // keep_top_k
                0,                                           // background_class
                op::v8::MatrixNms::DecayFunction::GAUSSIAN,  // decay_function
                2.0f,                                        // gaussian_sigma
                0.0f,                                        // post_threshold
                true,                                        // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {3, 6},
                                    std::vector<T_TH>{1.00,
                                                      0.95,
                                                      0.00,
                                                      0.00,
                                                      1.00,
                                                      1.00,
                                                      1.00,
                                                      0.8,
                                                      0.00,
                                                      10.00,
                                                      1.00,
                                                      11.00,
                                                      1.00,
                                                      0.1966116,
                                                      0.0,
                                                      0.1,
                                                      1.0,
                                                      1.1}),                       // expected_selected_scores
            reference_tests::Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}),           // expected_valid_outputs
            "matrix_nms_gaussian"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                0,                                         // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
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
            reference_tests::Tensor(ET_TH,
                                    {6, 6},
                                    std::vector<T_TH>{1.00,  0.95, 0.00,  0.00, 1.00,       1.00, 1.00, 0.8, 0.00,
                                                      10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,  1.0, 1.1,
                                                      1.00,  0.95, 0.00,  0.00, 1.00,       1.00, 1.00, 0.8, 0.00,
                                                      10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,  1.0, 1.1}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {6, 1}, std::vector<T_IND>{0, 3, 1, 6, 9, 7}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{3, 3}),                 // expected_valid_outputs
            "matrix_nms_two_batches_two_classes"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                true,                                      // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.5f,                                      // post_threshold
                true,                                      // normalized
            },
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
            "matrix_nms_two_batches_two_classes_by_score_cross_batch"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::CLASSID,  // sort_result_type
                true,                                        // sort_result_across_batch
                ET_IND,                                      // output_type
                0.0f,                                        // score_threshold
                3,                                           // nms_top_k
                -1,                                          // keep_top_k
                -1,                                          // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,    // decay_function
                2.0f,                                        // gaussian_sigma
                0.5f,                                        // post_threshold
                true,                                        // normalized
            },
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
            "matrix_nms_two_batches_two_classes_by_classid_cross_batch"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::CLASSID,  // sort_result_type
                false,                                       // sort_result_across_batch
                ET_IND,                                      // output_type
                0.0f,                                        // score_threshold
                3,                                           // nms_top_k
                3,                                           // keep_top_k
                0,                                           // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,    // decay_function
                2.0f,                                        // gaussian_sigma
                0.0f,                                        // post_threshold
                true,                                        // normalized
            },
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
            reference_tests::Tensor(ET_TH,
                                    {6, 6},
                                    std::vector<T_TH>{1.00,  0.95, 0.00,  0.00, 1.00,       1.00, 1.00, 0.8, 0.00,
                                                      10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,  1.0, 1.1,
                                                      1.00,  0.95, 0.00,  0.00, 1.00,       1.00, 1.00, 0.8, 0.00,
                                                      10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0,  0.1,  1.0, 1.1}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {6, 1}, std::vector<T_IND>{0, 3, 1, 6, 9, 7}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {2}, std::vector<T_IND>{3, 3}),                 // expected_valid_outputs
            "matrix_nms_by_keep_top_k"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 2, 6},
                std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {6, 6},
                std::vector<T_TH>{0.00, 0.95,       0.0, 10.0, 1.0, 11.0, 1.00, 0.95,       0.0, 0.0,  1.0, 1.0,
                                  0.00, 0.9,        0.0, 0.0,  1.0, 1.0,  1.00, 0.8,        0.0, 10.0, 1.0, 11.0,
                                  0.00, 0.13636364, 0.0, 0.1,  1.0, 1.1,  1.00, 0.13636364, 0.0, 0.1,  1.0, 1.1}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {6, 1}, std::vector<T_IND>{3, 0, 0, 3, 1, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{6}),                    // expected_valid_outputs
            "matrix_nms_background"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{1.0, 1.0,  0.0,  0.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  0.9, 1.0,  -0.1, 0.0,   10.0, 1.0,  11.0, 1.0, 10.1,
                                                                  0.0, 11.1, 1.0,  101.0, 0.0,  100.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {3, 6},
                                    std::vector<T_TH>{0.00,
                                                      0.95,
                                                      0.0,
                                                      10.0,
                                                      1.0,
                                                      11.0,
                                                      0.00,
                                                      0.9,
                                                      1.0,
                                                      1.0,
                                                      0.0,
                                                      0.0,
                                                      0.00,
                                                      0.75,
                                                      0.0,
                                                      0.1,
                                                      1.0,
                                                      1.1}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {3, 1}, std::vector<T_IND>{3, 0, 1}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{3}),           // expected_valid_outputs
            "matrix_nms_flipped_coordinates"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.8f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(
                ET_TH,
                {2, 6},
                std::vector<T_TH>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.9, 0.00, 0.00, 1.00, 1.00}),
            // expected_selected_scores
            reference_tests::Tensor(ET_IND, {2, 1}, std::vector<T_IND>{3, 0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{2}),        // expected_valid_outputs
            "matrix_nms_post_threshold"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.3f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 10, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                   0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                   0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                   0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                                                   0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}),  // boxes
            reference_tests::Tensor(
                ET_TH,
                {1, 1, 10},
                std::vector<T_TH>{0.4, 0.01, 0.2, 0.09, 0.15, 0.05, 0.02, 0.03, 0.05, 0.0}),  // scores
            reference_tests::Tensor(ET_TH,
                                    {1, 6},
                                    std::vector<T_TH>{0.00, 0.40, 0.00, 0.00, 1.00, 1.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}),     // expected_valid_outputs
            "matrix_nms_identical_boxes"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                2,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
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
            "matrix_nms_nms_top_k"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                0.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}),  // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 1}, std::vector<T_TH>{0.9}),           // scores
            reference_tests::Tensor(ET_TH,
                                    {1, 6},
                                    std::vector<T_TH>{0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),  // expected_selected_scores
            reference_tests::Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{1}),     // expected_valid_outputs
            "matrix_nms_single_box"),
        MatrixNmsParams(
            {
                op::v8::MatrixNms::SortResultType::SCORE,  // sort_result_type
                false,                                     // sort_result_across_batch
                ET_IND,                                    // output_type
                2.0f,                                      // score_threshold
                3,                                         // nms_top_k
                -1,                                        // keep_top_k
                -1,                                        // background_class
                op::v8::MatrixNms::DecayFunction::LINEAR,  // decay_function
                2.0f,                                      // gaussian_sigma
                0.0f,                                      // post_threshold
                true,                                      // normalized
            },
            reference_tests::Tensor(ET, {1, 6, 4}, std::vector<T>{0.0,  0.0,  1.0, 1.0,   0.0,  0.1,  1.0,  1.1, 0.0,
                                                                  -0.1, 1.0,  0.9, 0.0,   10.0, 1.0,  11.0, 0.0, 10.1,
                                                                  1.0,  11.1, 0.0, 100.0, 1.0,  101.0}),   // boxes
            reference_tests::Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),  // scores
            reference_tests::Tensor(ET_TH, {0, 6}, std::vector<T_TH>{}),    // expected_selected_scores
            reference_tests::Tensor(ET_IND, {0, 1}, std::vector<T_IND>{}),  // expected_selected_indices
            reference_tests::Tensor(ET_IND, {1}, std::vector<T_IND>{0}),    // expected_valid_outputs
            "matrix_nms_no_output"),
    };
    return params;
}

std::vector<MatrixNmsParams> generateCombinedParams() {
    const std::vector<std::vector<MatrixNmsParams>> generatedParams{
        generateParams<element::Type_t::bf16, element::Type_t::bf16, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::f16, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::bf16, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::f16, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<MatrixNmsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNms_With_Hardcoded_Refs,
                         ReferenceMatrixNmsTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceMatrixNmsTest::getTestCaseName);
}  // namespace
