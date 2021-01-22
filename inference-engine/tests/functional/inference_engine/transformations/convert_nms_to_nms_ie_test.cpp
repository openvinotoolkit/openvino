// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMSToNMSIEStatic) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE>(boxes, scores, max_output_boxes_per_class,
                std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                0, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMSToNMSIEDynamic1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();
        manager.run_passes(f);
        f->validate_nodes_and_infer_types();
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE>(boxes, scores, max_output_boxes_per_class,
                std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                0, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMSToNMSIEDynamic2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEMatcher>();
        manager.run_passes(f);
        f->validate_nodes_and_infer_types();
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE>(boxes, scores, max_output_boxes_per_class,
                std::make_shared<opset1::Unsqueeze>(iou_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                std::make_shared<opset1::Unsqueeze>(score_threshold, opset1::Constant::create(element::i64, Shape{1}, {0})),
                0, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMST1oNMSIE) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, op::v1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);
        auto convert = std::make_shared<opset1::Convert>(nms->output(0), element::i64);

        f_ref = std::make_shared<Function>(NodeVector{convert}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMST3oNMSIE) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMST4oNMSIE) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset4::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{1}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{1}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE3>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}