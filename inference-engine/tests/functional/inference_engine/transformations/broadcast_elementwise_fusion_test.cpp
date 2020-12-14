// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#define DYN ngraph::Dimension::dynamic()

using namespace testing;
using namespace ngraph;

using InputShape = PartialShape;
using TargetShape = Shape;

TEST(TransformationTests, BroadcastElementwiseFusion) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input_shape = InputShape{DYN, 3, 64, 64, 64};
        auto target_shape = TargetShape{8, 3, 64, 64, 64};
        auto input1 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, target_shape);
        auto target_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input1, target_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input2, broadcast);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto ref_target_shape1 = InputShape{8, 3, 64, 64, 64};
        auto ref_target_shape2 = InputShape{DYN, 3, 64, 64, 64};
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape1);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape2);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BroadcastElementwiseFusionSwitchInput) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input_shape = InputShape{DYN, 3, 64, 64, 64};
        auto target_shape = TargetShape{8, 3, 64, 64, 64};
        auto input1 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, target_shape);
        auto target_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input1, target_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(broadcast, input2);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto ref_target_shape1 = InputShape{DYN, 3, 64, 64, 64};
        auto ref_target_shape2 = InputShape{8, 3, 64, 64, 64};
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape1);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape2);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}