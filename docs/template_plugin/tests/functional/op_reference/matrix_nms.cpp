// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/constant.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct MatrixNmsParams {
    MatrixNmsParams(
        const int nms_top_k,
        const float score_threshold,
        const op::v8::MatrixNms::SortResultType sort_result_type,
        const int keep_top_k,
        const int background_class,
        const op::v8::MatrixNms::DecayFunction decay_function,
        const float gaussian_sigma,
        const float post_threshold,
        const ov::element::Type output_type,
        const bool sort_result_across_batch,
        const bool normalized,
        const Tensor& boxes, const Tensor& scores,
        const Tensor& expectedSelectedScores, const Tensor& expectedSelectedIndices,
        const Tensor& expectedValidOutputs, const std::string& testcaseName = "") :
        nms_top_k(nms_top_k),
        score_threshold(score_threshold),
        sort_result_type(sort_result_type),
        keep_top_k(keep_top_k),
        background_class(background_class),
        decay_function(decay_function),
        gaussian_sigma(gaussian_sigma),
        post_threshold(post_threshold),
        output_type(output_type),
        sort_result_across_batch(sort_result_across_batch),
        normalized(normalized),
        boxes(boxes), scores(scores),
        expectedSelectedScores(expectedSelectedScores), expectedSelectedIndices(expectedSelectedIndices),
        expectedValidOutputs(expectedValidOutputs), testcaseName(testcaseName) {}

    int nms_top_k = -1;
    float score_threshold = 0.0f;
    op::v8::MatrixNms::SortResultType sort_result_type = op::v8::MatrixNms::SortResultType::NONE;
    int keep_top_k = -1;
    int background_class = -1;
    op::v8::MatrixNms::DecayFunction decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    float gaussian_sigma = 2.0f;
    float post_threshold = 0.0f;
    ov::element::Type output_type = ov::element::i64;

    bool sort_result_across_batch = false;
    bool normalized = true;

    Tensor boxes;
    Tensor scores;
    Tensor expectedSelectedScores;
    Tensor expectedSelectedIndices;
    Tensor expectedValidOutputs;
    std::string testcaseName;
};

class ReferenceMatrixNmsTest : public testing::TestWithParam<MatrixNmsParams>, public CommonReferenceTest {
public:
    void SetUp() override {
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
    static std::shared_ptr<Function> CreateFunction(const MatrixNmsParams& params) {
        op::v8::MatrixNms::Attributes attrs;
        attrs.nms_top_k = params.nms_top_k;
        attrs.score_threshold = params.score_threshold;
        attrs.sort_result_type = params.sort_result_type;
        attrs.keep_top_k = params.keep_top_k;
        attrs.background_class = params.background_class;
        attrs.decay_function = params.decay_function;
        attrs.gaussian_sigma = params.gaussian_sigma;
        attrs.post_threshold = params.post_threshold;
        attrs.output_type = params.output_type;
        attrs.sort_result_across_batch = params.sort_result_across_batch;
        attrs.normalized = params.normalized;
        const auto boxes = std::make_shared<op::v0::Parameter>(params.boxes.type, PartialShape::dynamic());
        const auto scores = std::make_shared<op::v0::Parameter>(params.scores.type, PartialShape::dynamic());
        const auto nms = std::make_shared<op::v8::MatrixNms>(boxes, scores, attrs);
        const auto f = std::make_shared<Function>(nms->outputs(), ParameterVector{boxes, scores});
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
    std::vector<MatrixNmsParams> params {
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            0,                                          // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {3, 6}, std::vector<T_TH>{
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00,
                10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1}),             // expected_selected_scores
            Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),                        // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),                                 // expected_valid_outputs
            "matrix_nms_output_type_i64"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            0,                                          // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {3, 6}, std::vector<T_TH>{
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00,
                10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1}),             // expected_selected_scores
            Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),                        // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),                                 // expected_valid_outputs
            "matrix_nms_output_type_i32"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            0,                                          // background_class
            op::v8::MatrixNms::DecayFunction::GAUSSIAN, // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {3, 6}, std::vector<T_TH>{
                1.00,  0.95, 0.00,  0.00, 1.00,      1.00, 1.00, 0.8, 0.00,
                10.00, 1.00, 11.00, 1.00, 0.1966116, 0.0,  0.1,  1.0, 1.1}),            // expected_selected_scores
            Tensor(ET_IND, {3, 1}, std::vector<T_IND>{0, 3, 1}),                        // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),                                 // expected_valid_outputs
            "matrix_nms_gaussian"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            0,                                          // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {6, 6}, std::vector<T_TH>{
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1,
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {6, 1}, std::vector<T_IND>{0, 3, 1, 6, 9, 7}),               // expected_selected_indices
            Tensor(ET_IND, {2}, std::vector<T_IND>{3, 3}),                              // expected_valid_outputs
            "matrix_nms_two_batches_two_classes"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.5f,                                       // post_threshold
            ET_IND,                                     // output_type
            true,                                       // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {8, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00,
                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00,
                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                1.00, 0.80, 0.00, 10.00, 1.00, 11.00}),                                 // expected_selected_scores
            Tensor(ET_IND, {8, 1}, std::vector<T_IND>{3, 0, 9, 6, 0, 6, 3, 9}),         // expected_selected_indices
            Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),                              // expected_valid_outputs
            "matrix_nms_two_batches_two_classes_by_score_cross_batch"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::CLASSID, // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.5f,                                       // post_threshold
            ET_IND,                                     // output_type
            true,                                       // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {8, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00,
                0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00,
                0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                1.00, 0.80, 0.00, 10.00, 1.00, 11.00}),                                 // expected_selected_scores
            Tensor(ET_IND, {8, 1}, std::vector<T_IND>{3, 0, 9, 6, 0, 3, 6, 9}),         // expected_selected_indices
            Tensor(ET_IND, {2}, std::vector<T_IND>{4, 4}),                              // expected_valid_outputs
            "matrix_nms_two_batches_two_classes_by_classid_cross_batch"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::CLASSID, // sort_result_type
            3,                                          // keep_top_k
            0,                                          // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {2, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {2, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {6, 6}, std::vector<T_TH>{
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1,
                1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.8, 0.00, 10.00, 1.00, 11.00, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {6, 1}, std::vector<T_IND>{0, 3, 1, 6, 9, 7}),               // expected_selected_indices
            Tensor(ET_IND, {2}, std::vector<T_IND>{3, 3}),                              // expected_valid_outputs
            "matrix_nms_by_keep_top_k"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 2, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),      // scores
            Tensor(ET_TH, {6, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.0, 10.0, 1.0, 11.0, 1.00, 0.95,       0.0, 0.0, 1.0, 1.0, 0.00, 0.9,        0.0, 0.0, 1.0, 1.0,
                1.00, 0.8,  0.0, 10.0, 1.0, 11.0, 0.00, 0.13636364, 0.0, 0.1, 1.0, 1.1, 1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {6, 1}, std::vector<T_IND>{3, 0, 0, 3, 1, 1}),               // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{6}),                                 // expected_valid_outputs
            "matrix_nms_background"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
                0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}),   // boxes
            Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),                                       // scores
            Tensor(ET_TH, {3, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.0, 10.0, 1.0, 11.0, 0.00, 0.9, 1.0, 1.0, 0.0, 0.0, 0.00, 0.75, 0.0, 0.1, 1.0, 1.1}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {3, 1}, std::vector<T_IND>{3, 0, 1}),                        // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{3}),                                 // expected_valid_outputs
            "matrix_nms_flipped_coordinates"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.8f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),                                       // scores
            Tensor(ET_TH, {2, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.9, 0.00, 0.00, 1.00, 1.00}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {2, 1}, std::vector<T_IND>{3, 0}),                           // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{2}),                                 // expected_valid_outputs
            "matrix_nms_post_threshold"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.3f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 10, 4}, std::vector<T>{
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}),           // boxes
            Tensor(ET_TH, {1, 1, 10}, std::vector<T_TH>{
                0.4, 0.01, 0.2, 0.09, 0.15, 0.05, 0.02, 0.03, 0.05, 0.0}),              // scores
            Tensor(ET_TH, {1, 6}, std::vector<T_TH>{
                0.00, 0.40, 0.00, 0.00, 1.00, 1.00}),                                   // expected_selected_scores
            Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),                              // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{1}),                                 // expected_valid_outputs
            "matrix_nms_identical_boxes"),
        MatrixNmsParams(
            2,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),                                       // scores
            Tensor(ET_TH, {2, 6}, std::vector<T_TH>{
                0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),
                                                                                        // expected_selected_scores
            Tensor(ET_IND, {2, 1}, std::vector<T_IND>{3, 0}),                           // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{2}),                                 // expected_valid_outputs
            "matrix_nms_nms_top_k"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            0.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 1, 4}, std::vector<T>{0.0, 0.0, 1.0, 1.0}),                  // boxes
            Tensor(ET_TH, {1, 1, 1}, std::vector<T_TH>{0.9}),                           // scores
            Tensor(ET_TH, {1, 6}, std::vector<T_TH>{
                0.00, 0.90, 0.00, 0.00, 1.00, 1.00}),                                   // expected_selected_scores
            Tensor(ET_IND, {1, 1}, std::vector<T_IND>{0}),                              // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{1}),                                 // expected_valid_outputs
            "matrix_nms_single_box"),
        MatrixNmsParams(
            3,                                          // nms_top_k
            2.0f,                                       // score_threshold
            op::v8::MatrixNms::SortResultType::SCORE,   // sort_result_type
            -1,                                         // keep_top_k
            -1,                                         // background_class
            op::v8::MatrixNms::DecayFunction::LINEAR,   // decay_function
            2.0f,                                       // gaussian_sigma
            0.0f,                                       // post_threshold
            ET_IND,                                     // output_type
            false,                                      // sort_result_across_batch
            true,                                       // normalized
            Tensor(ET, {1, 6, 4}, std::vector<T>{
                0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),   // boxes
            Tensor(ET_TH, {1, 1, 6}, std::vector<T_TH>{
                0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),                                       // scores
            Tensor(ET_TH, {0, 6}, std::vector<T_TH>{}),                                 // expected_selected_scores
            Tensor(ET_IND, {0, 1}, std::vector<T_IND>{}),                               // expected_selected_indices
            Tensor(ET_IND, {1}, std::vector<T_IND>{0}),                                 // expected_valid_outputs
            "matrix_nms_no_output"),
    };
    return params;
}

std::vector<MatrixNmsParams> generateCombinedParams() {
    const std::vector<std::vector<MatrixNmsParams>> generatedParams {
        generateParams<element::Type_t::bf16, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<MatrixNmsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_MatrixNms_With_Hardcoded_Refs, ReferenceMatrixNmsTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceMatrixNmsTest::getTestCaseName);
} // namespace