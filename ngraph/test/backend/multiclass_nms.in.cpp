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
#include "engines_util/test_engines.hpp"
#include "engines_util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_score) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95,
                                                   0.00, 0.00, 1.00, 1.00,  0.00, 0.90,  0.00, 0.00,
                                                   1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00};
    std::vector<int64_t> expected_valid_outputs = {4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({4, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({4, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_class_id) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90,
                                                   0.00, 0.00, 1.00, 1.00,  1.00, 0.95,  0.00, 0.00,
                                                   1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00};
    std::vector<int64_t> expected_valid_outputs = {4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({4, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({4, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_output_type_i32) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;
    attrs.output_type = element::i32;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int32_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90,
                                                   0.00, 0.00, 1.00, 1.00,  1.00, 0.95,  0.00, 0.00,
                                                   1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00};
    std::vector<int32_t> expected_valid_outputs = {4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({4, 6}, expected_selected_scores);
    test_case.add_expected_output<int32_t>({4, 1}, expected_selected_indices);
    test_case.add_expected_output<int32_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_score) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3, 9, 6, 6, 9};
    std::vector<float> expected_selected_scores = {
        0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
        0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,  // 0
        0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
        0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00};  // 1
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_class_id) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3, 9, 6, 6, 9};
    std::vector<float> expected_selected_scores = {
        0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
        1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,  // 0
        0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
        1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00};  // 1
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_score_cross_batch) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;
    attrs.sort_result_across_batch = true;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6, 0, 6, 3, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   // 3
                                                   1.00, 0.95, 0.00, 0.00,  1.00, 1.00,    // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   // 9
                                                   1.00, 0.95, 0.00, 0.00,  1.00, 1.00,    // 6
                                                   0.00, 0.90, 0.00, 0.00,  1.00, 1.00,    // 0
                                                   0.00, 0.90, 0.00, 0.00,  1.00, 1.00,    // 6
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00,   // 3
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00};  // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_class_id_cross_batch) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;
    attrs.sort_result_across_batch = true;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6, 0, 3, 6, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   // 3
                                                   0.00, 0.90, 0.00, 0.00,  1.00, 1.00,    // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   // 9
                                                   0.00, 0.90, 0.00, 0.00,  1.00, 1.00,    // 6
                                                   1.00, 0.95, 0.00, 0.00,  1.00, 1.00,    // 0
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00,   // 3
                                                   1.00, 0.95, 0.00, 0.00,  1.00, 1.00,    // 6
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00};  // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({8, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({8, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_flipped_coordinates) {
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
                                     0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 1, M 6
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 1};
    std::vector<float> expected_selected_scores =
        {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 1.00, 1.00, 0.00, 0.00, 0.00, 0.75, 0.00, 0.10, 1.00, 1.10};
    std::vector<int64_t> expected_valid_outputs = {3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({3, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({3, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_identical_boxes) {
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 10, 4};  // N 1, C 1, M 10
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

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

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_limit_output_size) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 2;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores =
        {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({2, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({2, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_single_box) {
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

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

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_IOU) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.2f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores =
        {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({2, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({2, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_IOU_and_scores) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.95f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({1, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({1, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({1}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_no_output) {
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 2.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

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

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_background) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = 0;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {0, 3, 6, 9};
    std::vector<float> expected_selected_scores = {
        1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00,   // 0
        1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00};  // 1
    std::vector<int64_t> expected_valid_outputs = {2, 2};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({4, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({4, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_keep_top_k) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = 3;
    attrs.iou_threshold = 0.5f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = 3;
    attrs.background_class = -1;
    attrs.nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 9, 6, 6};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00,
                                                   0.00, 1.00, 1.00, 1.00,  0.95, 0.00,  0.00, 1.00, 1.00,  // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00,
                                                   0.00, 1.00, 1.00, 1.00,  0.95, 0.00,  0.00, 1.00, 1.00};  // 1
    std::vector<int64_t> expected_valid_outputs = {3, 3};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({6, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({6, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_nms_eta) {
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0,  // 0
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
        0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0  // 1
    };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,  // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3   // 1
    };

    op::v8::MulticlassNms::Attributes attrs;
    attrs.nms_top_k = -1;
    attrs.iou_threshold = 1.0f;
    attrs.score_threshold = 0.0f;
    attrs.sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = -1;
    attrs.background_class = -1;
    attrs.nms_eta = 0.1f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes, scores, attrs);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    std::vector<int64_t> expected_selected_indices = {3, 0, 5, 0, 3, 5, 9, 6, 11, 6, 9, 11};
    std::vector<float> expected_selected_scores = {
        0.00,   0.95, 0.00,   10.00,  1.00, 11.00,  0.00,   0.90, 0.00,   0.00,   1.00, 1.00,  0.00,  0.30, 0.00,
        100.00, 1.00, 101.00, 1.00,   0.95, 0.00,   0.00,   1.00, 1.00,   1.00,   0.80, 0.00,  10.00, 1.00, 11.00,
        1.00,   0.30, 0.00,   100.00, 1.00, 101.00, 0.00,   0.95, 0.00,   10.00,  1.00, 11.00, 0.00,  0.90, 0.00,
        0.00,   1.00, 1.00,   0.00,   0.30, 0.00,   100.00, 1.00, 101.00, 1.00,   0.95, 0.00,  0.00,  1.00, 1.00,
        1.00,   0.80, 0.00,   10.00,  1.00, 11.00,  1.00,   0.30, 0.00,   100.00, 1.00, 101.00};
    std::vector<int64_t> expected_valid_outputs = {6, 6};

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(f);
    test_case.add_multiple_inputs<float>({boxes_data, scores_data});
    test_case.add_expected_output<float>({12, 6}, expected_selected_scores);
    test_case.add_expected_output<int64_t>({12, 1}, expected_selected_indices);
    test_case.add_expected_output<int64_t>({2}, expected_valid_outputs);
    test_case.run();
}
