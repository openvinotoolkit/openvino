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
#include <numeric>
#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronDetectionOutput::Attributes;
using ExperimentalDO = op::v6::ExperimentalDetectronDetectionOutput;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_detection_output_eval)
{
    Attrs attrs;
    attrs.class_agnostic_box_regression = true;
    attrs.deltas_weights = {10.0f, 10.0f, 5.0f, 5.0f};
    attrs.max_delta_log_wh = 2.0f;
    attrs.max_detections_per_image = 5;
    attrs.nms_threshold = 0.2f;
    attrs.num_classes = 2;
    attrs.post_nms_count = 500;
    attrs.score_threshold = 0.01000000074505806f;

    auto rois = std::make_shared<op::Parameter>(element::f32, Shape{16, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{16, 8});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{16, 2});
    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{1, 3});

    auto detection = std::make_shared<ExperimentalDO>(rois, deltas, scores, im_info, attrs);

    auto f0 = make_shared<Function>(OutputVector{detection->output(0)}, ParameterVector{rois, deltas, scores, im_info});
    auto f1 = make_shared<Function>(OutputVector{detection->output(1)}, ParameterVector{rois, deltas, scores, im_info});
    auto f2 = make_shared<Function>(OutputVector{detection->output(2)}, ParameterVector{rois, deltas, scores, im_info});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> rois_data = {
        1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,  4.0f,  1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> deltas_data = {
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> scores_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> im_info_data = {1.0f, 1.0f, 1.0f};

    const auto output_boxes_shape = Shape{5, 4};
    const auto output_classes_shape = Shape{5};
    const auto output_scores_shape = Shape{5};

    std::vector<float> expected_output_boxes = {0.8929862f,
                                                0.892986297607421875,
                                                12.10701370239257812,
                                                12.10701370239257812,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f};

    std::vector<int32_t> expected_output_classes = {1, 0, 0, 0, 0};

    std::vector<float> expected_output_scores = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    auto output_boxes = backend->create_tensor(element::f32, output_boxes_shape);
    auto output_classes = backend->create_tensor(element::i32, output_classes_shape);
    auto output_scores = backend->create_tensor(element::f32, output_scores_shape);

    auto backend_rois = backend->create_tensor(element::f32, Shape{16, 4});
    auto backend_deltas = backend->create_tensor(element::f32, Shape{16, 8});
    auto backend_scores = backend->create_tensor(element::f32, Shape{16, 2});
    auto backend_im_info = backend->create_tensor(element::f32, Shape{1, 3});
    copy_data(backend_rois, rois_data);
    copy_data(backend_deltas, deltas_data);
    copy_data(backend_scores, scores_data);
    copy_data(backend_im_info, im_info_data);

    auto handle0 = backend->compile(f0);
    auto handle1 = backend->compile(f1);
    auto handle2 = backend->compile(f2);

    handle0->call_with_validate({output_boxes},
                                {backend_rois, backend_deltas, backend_scores, backend_im_info});
    handle1->call_with_validate({output_classes},
                                {backend_rois, backend_deltas, backend_scores, backend_im_info});
    handle2->call_with_validate({output_scores},
                                {backend_rois, backend_deltas, backend_scores, backend_im_info});

    auto output_boxes_vector = read_vector<float>(output_boxes);
    size_t output_boxes_shape_size = shape_size(output_boxes_shape);
    for (std::size_t j = 0; j < output_boxes_shape_size; ++j)
    {
        EXPECT_NEAR(output_boxes_vector[j], expected_output_boxes[j], 0.0000002);
    }

    EXPECT_EQ(expected_output_classes, read_vector<int32_t>(output_classes));

    EXPECT_TRUE(test::all_close_f(
        expected_output_scores, read_vector<float>(output_scores), MIN_FLOAT_TOLERANCE_BITS));
}
