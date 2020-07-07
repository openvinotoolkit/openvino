// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_nms_to_nms_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/nms_ie.hpp>
#include <ngraph/pass/constant_folding.hpp>

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
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        const auto & orig_shape = f->get_output_partial_shape(0);
        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMSToNMSIE().run_on_function(f);
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
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMSToNMSIE().run_on_function(f);
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
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMSToNMSIE().run_on_function(f);
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
