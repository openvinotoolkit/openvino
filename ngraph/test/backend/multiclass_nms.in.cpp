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
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_score)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};  // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,                                                    
                                                    1.00, 0.95, 0.00, 0.00, 1.00, 1.00 ,                                                    
                                                    0.00, 0.90, 0.00, 0.00, 1.00, 1.00,
                                                    1.00, 0.80, 0.00, 10.00, 1.00, 11.00 };
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_class_id)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                    0.00, 0.90, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.95, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.80, 0.00, 10.00, 1.00, 11.00  };
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_output_type_i32)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i32,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i32, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int32_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int32_t>(valid_outputs);

    std::vector<int32_t> expected_selected_indices = {3, 0, 0, 3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                    0.00, 0.90, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.95, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.80, 0.00, 10.00, 1.00, 11.00  };
    std::vector<int32_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_score)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3,
                                                      9, 6, 6, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00, // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 1
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_class_id)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1        
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 0, 3,
                                                      9, 6, 6, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00, // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 1
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_score_cross_batch)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);

    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  true,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6,
                                                      0, 6, 3, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   //3
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,  //0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //9
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00, //6                                           
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //0
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //6
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00, //3                                                   
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_two_batches_two_classes_by_class_id_cross_batch)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1        
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  true,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 9, 6,
                                                      0, 3, 6, 9};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00, //3
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   //9
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00, //6
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,   //0 
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00, // 3                                                   
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,    //6 
                                                   1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 9
    std::vector<int64_t> expected_valid_outputs = {4, 4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_flipped_coordinates)
{
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 1, M 6
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

   handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);                 

    std::vector<int64_t> expected_selected_indices = {3, 0, 1};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                   0.00, 0.90, 1.00, 1.00, 0.00, 0.00 ,
                                                   0.00, 0.75, 0.00, 0.10, 1.00, 1.10};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_identical_boxes)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 10, 4}; // N 1, C 1, M 10
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0};
    std::vector<float> expected_selected_scores = {0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_limit_output_size)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 2;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00 };
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_single_box)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0};
    std::vector<float> expected_selected_scores = {0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_IOU)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.2f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                   0.00, 0.90, 0.00, 0.00, 1.00, 1.00};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_IOU_and_scores)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.95f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_no_output)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 2.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::SCORE;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {};
    std::vector<float> expected_selected_scores = {};
    std::vector<int64_t> expected_valid_outputs = {0};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_background)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1        
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = 0;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 3, 6, 9};
    std::vector<float> expected_selected_scores = {1.00, 0.95, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00, // 0
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,     1.00, 0.80, 0.00, 10.00, 1.00, 11.00  }; // 1
    std::vector<int64_t> expected_valid_outputs = {2, 2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_keep_top_k)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1        
        };

    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = 3;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 0,
                                                      9, 6, 6};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00,     // 0
                                                   0.00, 0.95, 0.00, 10.00, 1.00, 11.00,   0.00, 0.90, 0.00, 0.00, 1.00, 1.00,
                                                   1.00, 0.95, 0.00, 0.00, 1.00, 1.00  };  // 1
    std::vector<int64_t> expected_valid_outputs = {3, 3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, multiclass_nms_by_nms_eta)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, // 0
                                     0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0 // 1                                   
                                     };

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3, // 0
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3 // 1        
        };

    const int64_t nms_top_k = -1;
    const float iou_threshold = 1.0f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 0.1f;

    const auto boxes_shape = Shape{2, 6, 4};  // N 2, C 2, M 6
    const auto scores_shape = Shape{2, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto nms = make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_outputs = backend->create_dynamic_tensor(element::f32, PartialShape::dynamic());
    auto selected_indeces = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());
    auto valid_outputs = backend->create_dynamic_tensor(element::i64, PartialShape::dynamic());

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_outputs, selected_indeces, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_scores_value = read_vector<float>(selected_outputs);
    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {3, 0, 5, 0, 3, 5, 
                                                      9, 6, 11, 6, 9, 11};
    std::vector<float> expected_selected_scores = {0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                    0.00, 0.90, 0.00, 0.00, 1.00, 1.00 ,
                                                    0.00, 0.30, 0.00, 100.00, 1.00, 101.00 ,
                                                    1.00, 0.95, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.80, 0.00, 10.00, 1.00, 11.00 ,
                                                    1.00, 0.30, 0.00, 100.00, 1.00, 101.00 ,
                                                    0.00, 0.95, 0.00, 10.00, 1.00, 11.00 ,
                                                    0.00, 0.90, 0.00, 0.00, 1.00, 1.00 ,
                                                    0.00, 0.30, 0.00, 100.00, 1.00, 101.00 ,
                                                    1.00, 0.95, 0.00, 0.00, 1.00, 1.00 ,
                                                    1.00, 0.80, 0.00, 10.00, 1.00, 11.00 ,
                                                    1.00, 0.30, 0.00, 100.00, 1.00, 101.00 };
    std::vector<int64_t> expected_valid_outputs = {6, 6};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}
