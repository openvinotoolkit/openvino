// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <tuple>
#include <openvino/opsets/opset8.hpp>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace reference_tests;

class ReferenceLoopTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
    }

public:
};

TEST_F(ReferenceLoopTest, CompareWithHardcodedRefs_dynamic_shape) {
    auto X = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Y = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Yi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M_body = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto body_condition = std::make_shared<ov::opset8::Constant>(ov::element::boolean, ov::Shape{1}, true);

    auto trip_count = std::make_shared<ov::opset8::Constant>(ngraph::element::i64, ov::Shape{1}, 3);
    auto exec_condition = std::make_shared<ov::opset8::Constant>(ngraph::element::boolean, ov::Shape{1}, true);
    // Body
    auto sum = std::make_shared<ov::opset8::Add>(Xi, Yi);
    auto Zo = std::make_shared<ov::opset8::Multiply>(sum, M_body);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo},
                                            ov::ParameterVector{Xi, Yi, M_body});

    auto loop = std::make_shared<ov::opset8::Loop>(trip_count, exec_condition);
    loop->set_function(body);

    loop->set_invariant_input(Xi, X);
    loop->set_invariant_input(Yi, Y);
    loop->set_merged_input(M_body, M, Zo);

    loop->set_special_body_ports(ov::opset8::Loop::SpecialBodyPorts{-1, 0});

    // Output is last Zo
    auto result = std::make_shared<ov::opset8::Result>(loop->get_iter_value(Zo, -1));
    function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{X, Y, M});

    auto input_shape = ov::Shape{2, 2};
    std::vector<float> inputX{0, 1, 2, 3}, inputY{1, 2, 3, 4}, inputM{5, 4, 3, 2};
    // 5*(0+1)*(0+1)*(0+1) = 5
    // 4*(1+2)*(1+2)*(1+2) = 108
    // 3*(2+3)*(2+3)*(2+3) = 375
    // 2*(3+4)*(3+4)*(3+4) = 686
    std::vector<float> expected_result{5, 108, 375, 686};
    Tensor inp_tensorX(ov::element::f32, input_shape, inputX);
    Tensor inp_tensorY(ov::element::f32, input_shape, inputY);
    Tensor inp_tensorM(ov::element::f32, input_shape, inputM);
    Tensor exp_tensor(ov::element::f32, input_shape, expected_result);
    inputData = {inp_tensorX.data, inp_tensorY.data, inp_tensorM.data};
    refOutData = {exp_tensor.data};

    Exec();
}
