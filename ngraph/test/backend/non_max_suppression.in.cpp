//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_center_point_box_format)
{
    std::vector<float> boxes_data = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,   1.0, 1.0,
                                     0.5, 0.4,  1.0, 1.0, 0.5, 10.5,  1.0, 1.0,
                                     0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CENTER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_flipped_coordinates)
{
    std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_identical_boxes)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                     0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                                     1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 10, 4};
    const auto scores_shape = Shape{1, 1, 10};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{1, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{1, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_limit_output_size)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{2, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{2, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_single_box)
{
    std::vector<float> boxes_data = {0.0, 0.0, 1.0, 1.0};

    std::vector<float> scores_data = {0.9};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 1, 4};
    const auto scores_shape = Shape{1, 1, 1};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{1, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{1, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {1};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_suppress_by_IOU)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{3, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{3, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 0.0, 0.3};
    std::vector<int64_t> expected_valid_outputs = {3};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_suppress_by_IOU_and_scores)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 3;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.4f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{2, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{2, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0};
    std::vector<float> expected_selected_scores = {0.0, 0.0, 0.95, 0.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {2};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_two_batches)
{
    std::vector<float> boxes_data = {
        0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
        0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
        0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{2, 6, 4};
    const auto scores_shape = Shape{2, 1, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{4, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{4, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 1, 0, 3, 1, 0, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 1.0, 0.0, 0.95, 1.0, 0.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}

NGRAPH_TEST(${BACKEND_NAME}, nonmaxsuppression_two_classes)
{
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t max_output_boxes_per_class_data = 2;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const auto box_encoding = op::v5::NonMaxSuppression::BoxEncodingType::CORNER;
    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 2, 6};

    const auto boxes = make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = make_shared<op::Parameter>(element::f32, scores_shape);
    auto max_output_boxes_per_class =
        op::Constant::create<int64_t>(element::i64, Shape{}, {max_output_boxes_per_class_data});
    auto iou_threshold = op::Constant::create<float>(element::f32, Shape{}, {iou_threshold_data});
    auto score_threshold =
        op::Constant::create<float>(element::f32, Shape{}, {score_threshold_data});
    auto soft_nms_sigma = op::Constant::create<float>(element::f32, Shape{}, {0.0f});
    auto nms = make_shared<op::v5::NonMaxSuppression>(boxes,
                                                      scores,
                                                      max_output_boxes_per_class,
                                                      iou_threshold,
                                                      score_threshold,
                                                      soft_nms_sigma,
                                                      box_encoding,
                                                      false);

    auto f = make_shared<Function>(nms, ParameterVector{boxes, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto selected_indeces = backend->create_tensor(element::i64, Shape{4, 3});
    auto selected_scores = backend->create_tensor(element::f32, Shape{4, 3});
    auto valid_outputs = backend->create_tensor(element::i64, Shape{1});

    auto backend_boxes = backend->create_tensor(element::f32, boxes_shape);
    auto backend_scores = backend->create_tensor(element::f32, scores_shape);
    copy_data(backend_boxes, boxes_data);
    copy_data(backend_scores, scores_data);

    auto handle = backend->compile(f);

    handle->call({selected_indeces, selected_scores, valid_outputs},
                 {backend_boxes, backend_scores});

    auto selected_indeces_value = read_vector<int64_t>(selected_indeces);
    auto selected_scores_value = read_vector<float>(selected_scores);
    auto valid_outputs_value = read_vector<int64_t>(valid_outputs);

    std::vector<int64_t> expected_selected_indices = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    std::vector<float> expected_selected_scores = {
        0.0, 0.0, 0.95, 0.0, 0.0, 0.9, 0.0, 1.0, 0.95, 0.0, 1.0, 0.9};
    std::vector<int64_t> expected_valid_outputs = {4};

    EXPECT_EQ(expected_selected_indices, selected_indeces_value);
    EXPECT_EQ(expected_selected_scores, selected_scores_value);
    EXPECT_EQ(expected_valid_outputs, valid_outputs_value);
}
