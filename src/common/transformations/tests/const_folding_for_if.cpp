// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
using namespace testing;
using namespace std;
using namespace ov;

// If doesn't have evaluate methods
TEST(TransformationTests, DISABLED_if_constant_folding) {
    std::shared_ptr<ov::Model> fun(nullptr);
    {
        auto cond = std::make_shared<opset5::Constant>(element::boolean, Shape{1}, false);
        auto A1 = std::make_shared<opset5::Constant>(element::f32, Shape{1}, 37.0);
        auto A2 = std::make_shared<opset5::Constant>(element::f32, Shape{1}, 45.0);
        auto B1 = std::make_shared<opset5::Constant>(element::f32, Shape{1}, 10.0);
        auto B2 = std::make_shared<opset5::Constant>(element::f32, Shape{1}, 3.0);
        auto Xt = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto Yt = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto Xe = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto Ye = make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
        auto a_add = std::make_shared<op::v1::Add>(Xt, Yt);
        auto b_pow = std::make_shared<op::v1::Power>(Xe, Ye);
        auto then_res = std::make_shared<op::v0::Result>(a_add);
        auto then_body = make_shared<ov::Model>(OutputVector{then_res}, ParameterVector{Xt, Yt});
        auto else_res = std::make_shared<op::v0::Result>(b_pow);
        auto else_body = make_shared<ov::Model>(OutputVector{else_res}, ParameterVector{Xe, Ye});
        auto if_op = make_shared<op::v8::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(A1, Xt, nullptr);
        if_op->set_input(A2, Yt, nullptr);
        if_op->set_input(B1, nullptr, Xe);
        if_op->set_input(B2, nullptr, Ye);
        auto if_res = if_op->set_output(then_res, else_res);
        auto param_add = make_shared<op::v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<op::v1::Add>(if_res, param_add);
        auto add_res = make_shared<op::v0::Result>(add);
        fun = make_shared<Model>(OutputVector{add_res}, ParameterVector{param_add});
        pass::Manager manager;
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(fun);
    }
    std::shared_ptr<ov::Model> f_ref(nullptr);
    {
        auto constant_folding_if = make_shared<opset5::Constant>(element::f32, Shape{1}, 1000.0f);
        auto param_add = make_shared<op::v0::Parameter>(element::f32, Shape{1});
        auto add = make_shared<op::v1::Add>(constant_folding_if, param_add);
        auto add_res = make_shared<op::v0::Result>(add);
        f_ref = std::make_shared<ov::Model>(NodeVector{add_res}, ParameterVector{param_add});
    }

    auto res = compare_functions(fun, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
