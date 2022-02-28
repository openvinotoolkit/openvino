// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset8.hpp"
#include "util/all_close_f.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, loop_dynamic_shapes) {
    // That which we iterate over
    auto X = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Y = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Yi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M_body = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::opset8::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);

    auto trip_count = std::make_shared<ov::opset8::Constant>(ngraph::element::i64, ngraph::Shape{1}, 3);
    auto exec_condition = std::make_shared<ov::opset8::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
    // Body
    auto sum = std::make_shared<ov::opset8::Add>(Xi, Yi);
    auto Zo = std::make_shared<ov::opset8::Multiply>(sum, M_body);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo}, ov::ParameterVector{Xi, Yi, M_body});

    auto loop = std::make_shared<ov::opset8::Loop>(trip_count, exec_condition);
    loop->set_function(body);

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    loop->set_special_body_ports(ov::opset8::Loop::SpecialBodyPorts{-1, 0});

    // Output is last Zo
    auto result = std::make_shared<ov::opset8::Result>(loop->get_iter_value(Zo, -1));
    auto f = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{X, Y, M});

    std::vector<float> inputX{0, 1, 2, 3}, inputY{1, 2, 3, 4}, inputM{5, 4, 3, 2};
    std::vector<float> expected_result{5, 108, 375, 686};
    std::vector<float> actual_result(ov::shape_size(ov::Shape{2, 2}), 2);

    auto r0 = std::make_shared<ov::HostTensor>();
    using namespace ngraph;
    ASSERT_TRUE(f->evaluate({r0},
                            {make_host_tensor<ngraph::element::Type_t::f32>(ov::Shape{2, 2}, inputX),
                             make_host_tensor<ngraph::element::Type_t::f32>(ov::Shape{2, 2}, inputY),
                             make_host_tensor<ngraph::element::Type_t::f32>(ov::Shape{2, 2}, inputM)}));

    EXPECT_EQ(r0->get_shape(), (ov::Shape{2, 2}));
    memcpy(actual_result.data(), r0->get_data_ptr<float>(), ov::shape_size(ov::Shape{2, 2}) * sizeof(float));
    EXPECT_TRUE(ngraph::test::all_close_f(expected_result, actual_result));
}

TEST(op_eval, loop_dynamic_output) {
    // That which we iterate over
    auto X = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 2});
    auto T = std::make_shared<ov::opset8::Parameter>(ov::element::i64, ov::Shape{});

    // Set up the cell body, a function from (Xi) -> Concat(Xi, C) -> (Zo)
    // Body parameters
    auto Xi =
        std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 2});

    auto body_condition = std::make_shared<ov::opset8::Constant>(ov::element::boolean, ov::Shape{}, true);
    auto exec_condition = std::make_shared<ov::opset8::Constant>(ov::element::boolean, ov::Shape{}, true);

    // Body
    auto C = ov::opset8::Constant::create(ov::element::f32, {1, 1, 2}, {2});
    auto Zo = std::make_shared<ov::opset8::Concat>(ov::NodeVector{Xi, C}, 1);
    auto Z = std::make_shared<ov::opset8::Result>(Zo);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{Z, body_condition}, ov::ParameterVector{Xi});

    auto loop = std::make_shared<ov::opset8::Loop>(T, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::opset8::Loop::SpecialBodyPorts{-1, 1});

    loop->set_merged_input(Xi, X, Z);
    // Output 1 is last Z
    auto out1 = loop->get_iter_value(Z, -1);
    auto result1 = std::make_shared<ov::opset8::Result>(out1);

    auto results = ov::ResultVector{result1};
    auto f = std::make_shared<ov::Model>(results, ov::ParameterVector{X, T});
    std::vector<float> inputX{0, 0};
    std::vector<float> expected_result{0, 0, 2, 2, 2, 2, 2, 2};
    std::vector<float> actual_result(ov::shape_size(ov::Shape{1, 4, 2}));

    auto r0 = std::make_shared<ov::HostTensor>();
    using namespace ngraph;
    ASSERT_TRUE(f->evaluate({r0},
                            {make_host_tensor<ngraph::element::Type_t::f32>(ov::Shape{1, 1, 2}, inputX),
                             make_host_tensor<ngraph::element::Type_t::i64>(ov::Shape{}, {3})}));

    EXPECT_EQ(r0->get_shape(), (ov::Shape{1, 4, 2}));
    memcpy(actual_result.data(), r0->get_data_ptr<float>(), ov::shape_size(ov::Shape{1, 4, 2}) * sizeof(float));
    EXPECT_TRUE(ngraph::test::all_close_f(expected_result, actual_result));
}