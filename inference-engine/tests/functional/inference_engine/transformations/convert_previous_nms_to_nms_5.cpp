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
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/op_conversions/convert_previous_nms_to_nms_5.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMS4FiveInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset4::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset4::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset4::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset4::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS4ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS4TwoInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset4::NonMaxSuppression>(boxes, scores, opset4::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS4ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS3FiveInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS3ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS3TwoInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS3ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS1FiveInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset1::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset1::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset1::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
                                                               opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS1ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertNMS1TwoInputsToNMS5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ConvertNMS1ToNMS5>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset5::Constant::create(element::i64, Shape{}, {0});
        auto iou_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto score_threshold = opset5::Constant::create(element::f32, Shape{}, {0.0f});
        auto nms = std::make_shared<opset5::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,  iou_threshold, score_threshold,
                                                               opset5::NonMaxSuppression::BoxEncodingType::CORNER, true);

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
