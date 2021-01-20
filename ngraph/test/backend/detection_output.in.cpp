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
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, detection_output_3_inputs)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 3;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = false;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        1, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto f = make_shared<Function>(make_shared<op::DetectionOutput>(loc, conf, prior_boxes, attrs),
                                   ParameterVector{loc, conf, prior_boxes});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0, class 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 0, class 1
        0.3,
        0.2,
        0.5,
        0.3,
        0.2,
        0.1,
        0.42,
        0.66,
        // batch 0, class 2
        0.05,
        0.1,
        0.2,
        0.3,
        0.2,
        0.1,
        0.33,
        0.44,
        // batch 1, class 0
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
        // batch 1, class 1
        0.1,
        0.2,
        0.5,
        0.3,
        0.1,
        0.1,
        0.12,
        0.34,
        // batch 1, class 2
        0.25,
        0.11,
        0.4,
        0.32,
        0.2,
        0.12,
        0.38,
        0.24,
    });
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        0,
        0.2,
        // batch 1
        0.7,
        0.8,
        0.42,
        0.33,
        0.81,
        0.2,
    });
    test_case.add_input<float>({
        // prior box 0
        0.0,
        0.5,
        0.1,
        0.2,
        // prior box 1
        0.0,
        0.3,
        0.1,
        0.35,
    });
    Shape output_shape{1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7};
    test_case.add_expected_output<float>(
        output_shape, {0, 0, 0.7,  0.2,  0.4,  0.52, 1,    0, 1, 0.9, 0,   0.6,  0.3, 0.35,
                       1, 1, 0.81, 0.25, 0.41, 0.5,  0.67, 1, 1, 0.8, 0.1, 0.55, 0.3, 0.45});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, detection_output_3_inputs_share_location)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 3;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = true;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto f = make_shared<Function>(make_shared<op::DetectionOutput>(loc, conf, prior_boxes, attrs),
                                   ParameterVector{loc, conf, prior_boxes});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 1
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
    });
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        0,
        0.2,
        // batch 1
        0.7,
        0.8,
        0.42,
        0.33,
        0.81,
        0.2,
    });
    test_case.add_input<float>({
        // batch 0
        0.0,
        0.5,
        0.1,
        0.2,
        0.0,
        0.3,
        0.1,
        0.35,
        // batch 1
        0.33,
        0.2,
        0.52,
        0.37,
        0.22,
        0.1,
        0.32,
        0.36,
    });
    Shape output_shape{1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7};
    test_case.add_expected_output<float>(output_shape,
                                         {
                                             0,    0,   0.7, 0,   0.4,  0.3, 0.5,  0,    1,    0.9,
                                             0.1,  0.6, 0.3, 0.4, 1,    1,   0.81, 0.32, 0.15, 0.52,
                                             0.61, 1,   1,   0.8, 0.53, 0.3, 0.92, 0.57,

                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, detection_output_3_inputs_normalized)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 3;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = true;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto f = make_shared<Function>(make_shared<op::DetectionOutput>(loc, conf, prior_boxes, attrs),
                                   ParameterVector{loc, conf, prior_boxes});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 1
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
    });
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        0,
        0.2,
        // batch 1
        0.7,
        0.8,
        0.42,
        0.33,
        0.81,
        0.2,
    });
    test_case.add_input<float>({
        // batch 0
        0.0,
        0.5,
        0.1,
        0.2,
        0.0,
        0.3,
        0.1,
        0.35,
        // batch 1
        0.33,
        0.2,
        0.52,
        0.37,
        0.22,
        0.1,
        0.32,
        0.36,
    });
    Shape output_shape{1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7};
    test_case.add_expected_output<float>(output_shape,
                                         {
                                             0,    0,   0.7, 0,   0.4,  0.3, 0.5,  0,    1,    0.9,
                                             0.1,  0.6, 0.3, 0.4, 1,    1,   0.81, 0.32, 0.15, 0.52,
                                             0.61, 1,   1,   0.8, 0.53, 0.3, 0.92, 0.57,

                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, detection_output_3_inputs_keep_all_bboxes)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 2;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = false;
    attrs.keep_top_k = {-1};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = false;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 3;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto f = make_shared<Function>(make_shared<op::DetectionOutput>(loc, conf, prior_boxes, attrs),
                                   ParameterVector{loc, conf, prior_boxes});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0, class 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 0, class 1
        0.3,
        0.2,
        0.5,
        0.3,
        0.2,
        0.1,
        0.42,
        0.66,
        // batch 1, class 0
        0.05,
        0.1,
        0.2,
        0.3,
        0.2,
        0.1,
        0.33,
        0.44,
        // batch 1, class 1
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
        // batch 2, class 0
        0.1,
        0.2,
        0.5,
        0.3,
        0.1,
        0.1,
        0.12,
        0.34,
        // batch 2, class 1
        0.25,
        0.11,
        0.4,
        0.32,
        0.2,
        0.12,
        0.38,
        0.24,
    });
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        // batch 1
        0.7,
        0.8,
        0.42,
        0.33,
        // batch 1
        0.1,
        0.2,
        0.32,
        0.43,
    });
    test_case.add_input<float>({
        // batch 0 priors
        0.0,
        0.5,
        0.1,
        0.2,
        0.0,
        0.3,
        0.1,
        0.35,
        // batch 0 variances
        0.12,
        0.11,
        0.32,
        0.02,
        0.02,
        0.20,
        0.09,
        0.71,
        // batch 1 priors
        0.33,
        0.2,
        0.52,
        0.37,
        0.22,
        0.1,
        0.32,
        0.36,
        // batch 1 variances
        0.01,
        0.07,
        0.12,
        0.13,
        0.41,
        0.33,
        0.2,
        0.1,
        // batch 2 priors
        0.0,
        0.3,
        0.1,
        0.35,
        0.22,
        0.1,
        0.32,
        0.36,
        // batch 2 variances
        0.32,
        0.02,
        0.13,
        0.41,
        0.33,
        0.2,
        0.02,
        0.20,
    });
    Shape output_shape{1, 1, num_images * attrs.num_classes * num_prior_boxes, 7};
    test_case.add_expected_output<float>(
        output_shape,
        {

            0, 0, 0.4,  0.006, 0.34,   0.145,  0.563,  0,  1, 0.9,  0,      0.511, 0.164,  0.203,
            0, 1, 0.7,  0.004, 0.32,   0.1378, 0.8186, 1,  0, 0.7,  0.3305, 0.207, 0.544,  0.409,
            1, 0, 0.42, 0.302, 0.133,  0.4,    0.38,   1,  1, 0.8,  0.332,  0.207, 0.5596, 0.4272,
            1, 1, 0.33, 0.261, 0.1165, 0.36,   0.385,  2,  0, 0.32, 0.3025, 0.122, 0.328,  0.424,
            2, 1, 0.43, 0.286, 0.124,  0.3276, 0.408,  -1, 0, 0,    0,      0,     0,      0,
            0, 0, 0,    0,     0,      0,      0,      0,  0, 0,    0,      0,     0,      0,

        });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, detection_output_3_inputs_center_size)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 3;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CENTER_SIZE";
    attrs.share_location = false;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto f = make_shared<Function>(make_shared<op::DetectionOutput>(loc, conf, prior_boxes, attrs),
                                   ParameterVector{loc, conf, prior_boxes});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0, class 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 0, class 1
        0.3,
        0.2,
        0.5,
        0.3,
        0.2,
        0.1,
        0.42,
        0.66,
        // batch 0, class 2
        0.05,
        0.1,
        0.2,
        0.3,
        0.2,
        0.1,
        0.33,
        0.44,
        // batch 1, class 0
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
        // batch 1, class 1
        0.1,
        0.2,
        0.5,
        0.3,
        0.1,
        0.1,
        0.12,
        0.34,
        // batch 1, class 2
        0.25,
        0.11,
        0.4,
        0.32,
        0.2,
        0.12,
        0.38,
        0.24,
    });
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        0,
        0.2,
        // batch 1
        0.7,
        0.8,
        0.42,
        0.33,
        0.81,
        0.2,
    });
    test_case.add_input<float>({
        // batch 0
        0.0,
        0.5,
        0.1,
        0.2,
        0.0,
        0.3,
        0.1,
        0.35,
        // batch 1
        0.33,
        0.2,
        0.52,
        0.37,
        0.22,
        0.1,
        0.32,
        0.36,
    });
    Shape output_shape{1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7};
    test_case.add_expected_output<float>(
        output_shape,
        {
            0, 0, 0.7,  0,          0.28163019,  0.14609808, 0.37836978,
            0, 1, 0.9,  0,          0.49427515,  0.11107014, 0.14572485,
            1, 1, 0.81, 0.22040875, 0.079573378, 0.36959124, 0.4376266,
            1, 1, 0.8,  0.32796675, 0.18435785,  0.56003326, 0.40264216,
        });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, detection_output_5_inputs)
{
    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 2;
    attrs.background_label_id = -1;
    attrs.top_k = -1;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {2};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = false;
    attrs.nms_threshold = 0.5;
    attrs.confidence_threshold = 0.3;
    attrs.clip_after_nms = false;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 0;
    attrs.input_width = 0;
    attrs.objectness_score = 0.6;

    size_t num_prior_boxes = 2;
    size_t num_loc_classes = attrs.share_location ? 1 : attrs.num_classes;
    size_t prior_box_size = attrs.normalized ? 4 : 5;
    size_t num_images = 2;
    Shape loc_shape{num_images, num_prior_boxes * num_loc_classes * prior_box_size};
    Shape conf_shape{num_images, num_prior_boxes * attrs.num_classes};
    Shape prior_boxes_shape{
        num_images, attrs.variance_encoded_in_target ? 1UL : 2UL, num_prior_boxes * prior_box_size};

    auto loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto prior_boxes = make_shared<op::Parameter>(element::f32, prior_boxes_shape);
    auto aux_loc = make_shared<op::Parameter>(element::f32, loc_shape);
    auto aux_conf = make_shared<op::Parameter>(element::f32, conf_shape);
    auto f = make_shared<Function>(
        make_shared<op::DetectionOutput>(loc, conf, prior_boxes, aux_conf, aux_loc, attrs),
        ParameterVector{loc, conf, prior_boxes, aux_conf, aux_loc});

    auto test_case = test::TestCase<TestEngine>(f);
    // locations
    test_case.add_input<float>({
        // batch 0, class 0
        0.1,
        0.1,
        0.2,
        0.2,
        0.0,
        0.1,
        0.2,
        0.15,
        // batch 0, class 1
        0.3,
        0.2,
        0.5,
        0.3,
        0.2,
        0.1,
        0.42,
        0.66,
        // batch 1, class 0
        0.2,
        0.1,
        0.4,
        0.2,
        0.1,
        0.05,
        0.2,
        0.25,
        // batch 1, class 1
        0.1,
        0.2,
        0.5,
        0.3,
        0.1,
        0.1,
        0.12,
        0.34,
    });
    // confidence
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.9,
        0.4,
        0.7,
        // batch 1
        0.42,
        0.33,
        0.81,
        0.2,
    });
    // prior boxes
    test_case.add_input<float>({
        // batch 0
        0.0,
        0.5,
        0.1,
        0.2,
        0.0,
        0.3,
        0.1,
        0.35,
        // batch 1
        0.33,
        0.2,
        0.52,
        0.37,
        0.22,
        0.1,
        0.32,
        0.36,
    });
    // aux conf
    test_case.add_input<float>({
        // batch 0
        0.1,
        0.3,
        0.5,
        0.8,
        // batch 1
        0.5,
        0.8,
        0.01,
        0.1,
    });
    // aux locations
    test_case.add_input<float>({
        // batch 0, class 0
        0.1,
        0.2,
        0.5,
        0.3,
        0.1,
        0.1,
        0.12,
        0.34,
        // batch 0, class 1
        0.25,
        0.11,
        0.4,
        0.32,
        0.2,
        0.12,
        0.38,
        0.24,
        // batch 1, class 0
        0.3,
        0.2,
        0.5,
        0.3,
        0.2,
        0.1,
        0.42,
        0.66,
        // batch 1, class 1
        0.05,
        0.1,
        0.2,
        0.3,
        0.2,
        0.1,
        0.33,
        0.44,
    });

    Shape output_shape{1, 1, num_images * static_cast<size_t>(attrs.keep_top_k[0]), 7};
    test_case.add_expected_output<float>(
        output_shape,
        {
            0, 0, 0.4,  0.55, 0.61, 1, 0.97, 0, 1, 0.7,  0.4,  0.52, 0.9, 1,
            1, 0, 0.42, 0.83, 0.5,  1, 0.87, 1, 1, 0.33, 0.63, 0.35, 1,   1,

        });
    test_case.run();
}
