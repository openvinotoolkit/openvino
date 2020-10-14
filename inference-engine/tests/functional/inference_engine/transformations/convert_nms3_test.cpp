// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/op_conversions/convert_nms3.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvertNMS3I32Output) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset1::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset1::NonMaxSuppression::BoxEncodingType::CORNER, true);
        nms->set_friendly_name("nms");

        f = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        pass::InitNodeInfo().run_on_function(f);
        pass::ConvertNMS1ToNMS3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto boxes = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset3::Parameter>(element::f32, Shape{1, 1, 1000});
        auto max_output_boxes_per_class = opset3::Constant::create(element::i64, Shape{}, {10});
        auto iou_threshold = opset3::Constant::create(element::f32, Shape{}, {0.75});
        auto score_threshold = opset3::Constant::create(element::f32, Shape{}, {0.7});
        auto nms = std::make_shared<opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                iou_threshold, score_threshold, opset3::NonMaxSuppression::BoxEncodingType::CORNER, true);
        nms->set_friendly_name("nms");

        f_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto nms_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(nms_node->get_friendly_name() == "nms") << "Transformation ConvertNMS1ToNMS3 should keep output names.\n";
}
