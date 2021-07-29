// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_output_type_i64)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = 0;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0, 3, 1};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({3, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({3, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_output_type_i32)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = 0;

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;
    attrs.output_type = ngraph::element::i32;

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int32_t> expected_selected_indices = {0, 3, 1};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};
    std::vector<int32_t> expected_valid_outputs = {3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({3, 6}, expected_selected_scores);
    test_case.add_expected_output<int32_t>({3, 1}, expected_selected_indices);
    test_case.add_expected_output<int32_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_gaussian)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = 0;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::GAUSSIAN;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0, 3, 1};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.1966116, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({3, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({3, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_two_batches_two_classes)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}; // 1

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}; // 1

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = 0;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0, 3, 1,
                                                      6, 9, 7};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {3, 3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({6, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({6, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_two_batches_two_classes_by_score_cross_batch)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}; // 1

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}; // 1

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.5;
    attrs.sort_result_across_batch = true;

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6,
                                                      0, 6, 3, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //3
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00, //0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //9
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00, //6
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //0
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //6
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00, //3
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00}; // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_two_batches_two_classes_by_classid_cross_batch)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}; // 1

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}; // 1

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.5;
    attrs.sort_result_across_batch = true;

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6,
                                                      0, 3, 6, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //3
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //9
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //6
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00, //0
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00, // 3
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00, //6
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_by_keep_top_k)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}; // 1

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3}; // 1

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::CLASSID;
    attrs.keep_top_k = 3;
    attrs.background_class = 0;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0, 3, 1,
                                                      6, 9, 7};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.8, 0.00, 10.00, 1.00, 11.00,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {3, 3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({6, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({6, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_background)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3, 1, 1};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.0, 10.0, 1.0, 11.0,
                                                   1.00, 0.95, 0.0, 0.0, 1.0, 1.0,
                                                   0.00, 0.9, 0.0, 0.0, 1.0, 1.0,
                                                   1.00, 0.8, 0.0, 10.0, 1.0, 11.0,
                                                   0.00, 0.13636364, 0.0, 0.1, 1.0, 1.1,
                                                   1.00, 0.13636364, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {6};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({6, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({6, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_flipped_coordinates)
{
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 1, M 6
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 1};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.0, 10.0, 1.0, 11.0,
                                                   0.00, 0.9, 1.0, 1.0, 0.0, 0.0,
                                                   0.00, 0.75, 0.0, 0.1, 1.0, 1.1};
    std::vector<int64_t> expected_valid_outputs = {3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({3, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({3, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_post_threshold)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.00;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.8;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,
                                                   0.00, 0.9, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({2, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({2, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_identical_boxes)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.4, 0.01, 0.2, 0.09, 0.15, 0.05, 0.02, 0.03, 0.05, 0.0};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 10, 4}; // N 1, C 1, M 10
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.3;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0};
    std::vector<float> expected_selected_scores = {0.00, 0.40, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({1, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({1, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_nms_top_k)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 2;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00 };
    std::vector<int64_t> expected_valid_outputs = {2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({2, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({2, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_single_box)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0};
    std::vector<float> expected_selected_scores = {0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({1, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({1, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, matrix_nms_no_output)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MatrixNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.score_threshold = 2.0f;
    attrs.sort_result_type = op::v8::MatrixNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    attrs.decay_function = op::v8::MatrixNms::DecayFunction::LINEAR;
    attrs.gaussian_sigma = 2.0f;
    attrs.post_threshold = 0.0f;

    auto nms = make_shared<op::v8::MatrixNms>(boxes, scores, attrs);


    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {};
    std::vector<float> expected_selected_scores = {};
    std::vector<int64_t> expected_valid_outputs = {0};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({0, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({0, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}
