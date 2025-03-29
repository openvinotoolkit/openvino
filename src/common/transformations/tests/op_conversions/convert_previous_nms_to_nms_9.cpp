// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertNMS5SixInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_sigma = opset5::Constant::create(element::f32, Shape{}, {0.1});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.1});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS5TwoInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes,
                                                               scores,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS5ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS4FiveInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS4TwoInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS4ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS3FiveInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS3TwoInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS3ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS1FiveInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMS1TwoInputsToNMS9) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes,
                                                               scores,
                                                               opset1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertNMS1ToNMS9>();
    }

    {
        auto boxes = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset9::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset9::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto soft_sigma = opset9::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset9::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               soft_sigma,
                                                               opset9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}
