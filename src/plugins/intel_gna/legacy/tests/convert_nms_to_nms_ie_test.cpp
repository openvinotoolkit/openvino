// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/ngraph_ops/nms_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertNMSToNMSIEStatic) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();

        // as inside test infrastructure we can not predict output names for given Function
        // we have to enable soft names comparison manually
        enable_soft_names_comparison();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE>(
            boxes,
            scores,
            max_output_boxes_per_class,
            std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            0,
            true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMSToNMSIEDynamic1) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE>(
            boxes,
            scores,
            max_output_boxes_per_class,
            std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            0,
            true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMSToNMSIEDynamic2) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE>(
            boxes,
            scores,
            max_output_boxes_per_class,
            std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
            0,
            true);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMST1oNMSIE) {
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
                                                               op::v1::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE3>(boxes,
                                                                      scores,
                                                                      max_output_boxes_per_class,
                                                                      iou_threshold,
                                                                      score_threshold,
                                                                      0,
                                                                      true,
                                                                      element::i32);
        auto convert = std::make_shared<opset1::Convert>(nms->output(0), element::i64);

        model_ref = std::make_shared<Model>(NodeVector{convert}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMST3oNMSIE) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true,
                                                               element::i32);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE3>(boxes,
                                                                      scores,
                                                                      max_output_boxes_per_class,
                                                                      iou_threshold,
                                                                      score_threshold,
                                                                      0,
                                                                      true,
                                                                      element::i32);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

TEST_F(TransformationTestsF, ConvertNMST4oNMSIE) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes,
                                                               scores,
                                                               max_output_boxes_per_class,
                                                               iou_threshold,
                                                               score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true,
                                                               element::i32);

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<ngraph::op::NonMaxSuppressionIE3>(boxes,
                                                                      scores,
                                                                      max_output_boxes_per_class,
                                                                      iou_threshold,
                                                                      score_threshold,
                                                                      0,
                                                                      true,
                                                                      element::i32);

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}
