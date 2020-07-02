// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph_ops/nms_ie.hpp>
#include <transformations/convert_opset1_to_legacy/convert_nms_4_to_legacy.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMS4ToNMSIEStatic) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        const auto &orig_shape = f->get_output_partial_shape(0);
        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMS4ToLegacy().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE2>(boxes, scores, max_output_boxes_per_class,
                                                              std::make_shared<opset4::Unsqueeze>(iou_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              std::make_shared<opset4::Unsqueeze>(score_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              0, true);
        auto convert = std::make_shared<ngraph::opset4::Convert>(nms, element::i64);
        convert->set_friendly_name("nms");

        f_ref = std::make_shared<Function>(NodeVector{convert}, ParameterVector{boxes, scores});
        ASSERT_TRUE(f_ref->get_output_partial_shape(0).is_static()) << "Shape " << f_ref->get_output_partial_shape(0) << " should be static";
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS4ToNMSIEDynamic1) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMS4ToLegacy().run_on_function(f);
        f->validate_nodes_and_infer_types();
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic());
        auto scores = std::make_shared<opset4::Parameter>(element::f32, PartialShape::dynamic());
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE2>(boxes, scores, max_output_boxes_per_class,
                                                              std::make_shared<opset4::Unsqueeze>(iou_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              std::make_shared<opset4::Unsqueeze>(score_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              0, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS4ToNMSIEDynamic2) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER, true, element::i32);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMS4ToLegacy().run_on_function(f);
        f->validate_nodes_and_infer_types();
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, PartialShape{DYN, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, PartialShape{DYN, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{1}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<op::NonMaxSuppressionIE2>(boxes, scores, max_output_boxes_per_class,
                                                              std::make_shared<opset4::Unsqueeze>(iou_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              std::make_shared<opset4::Unsqueeze>(score_threshold,
                                                                                                  opset4::Constant::create(element::i64, Shape{1}, {0})),
                                                              0, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
