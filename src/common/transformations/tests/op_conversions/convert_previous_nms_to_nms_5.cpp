// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_5.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertNMS4FiveInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS4ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS4TwoInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<op::v4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               op::v4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS4ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS3FiveInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS3ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS3TwoInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<op::v3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               op::v3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS3ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS1FiveInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v1::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS1ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS1TwoInputsToNMS5) {
    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<op::v1::NonMaxSuppression>(boxes,
                                                               scores,
                                                               op::v1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS1ToNMS5>();
    }

    {
        auto boxes = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = op::v0::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<op::v5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               op::v5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}
