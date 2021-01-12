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
#include <ngraph/opsets/opset5.hpp>
#include <transformations/op_conversions/convert_nms_to_nms_ie_internal.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/nms_ie_internal.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMS1ToNMSIEInternal) {
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

        const auto & orig_shape = f->get_output_partial_shape(0);
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMS1ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
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
        auto nms = std::make_shared<op::internal::NonMaxSuppressionIEInternal>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);
        auto convert = std::make_shared<opset1::Convert>(nms->output(0), element::i64);

        f_ref = std::make_shared<Function>(NodeVector{convert}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS3ToNMSIEInternal) {
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

        const auto & orig_shape = f->get_output_partial_shape(0);
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMS3ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
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
        auto nms = std::make_shared<op::internal::NonMaxSuppressionIEInternal>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS4ToNMSIEInternal) {
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

        const auto & orig_shape = f->get_output_partial_shape(0);
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMS4ToNMS5>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
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
        auto nms = std::make_shared<op::internal::NonMaxSuppressionIEInternal>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, 0, true, element::i32);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS5ToNMSIEInternal) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i32, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto soft_nms_sigma = opset1::Constant::create(element::f32, Shape{}, {0.5});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, soft_nms_sigma, opset5::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        const auto & orig_shape = f->get_output_partial_shape(0);
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertNMSToNMSIEInternal>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
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
        auto soft_nms_sigma = opset1::Constant::create(element::f32, Shape{1}, {0.5});
        auto nms = std::make_shared<op::internal::NonMaxSuppressionIEInternal>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, soft_nms_sigma, 0, true, element::i32);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
