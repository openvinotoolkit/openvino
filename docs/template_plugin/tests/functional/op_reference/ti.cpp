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

class ReferenceTensorIteratorTest : public testing::Test, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
    }

public:
};

TEST_F(ReferenceTensorIteratorTest, CompareWithHardcodedRefs_dynamic_shape) {
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

    auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();
    tensor_iterator->set_function(body);

    tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
    tensor_iterator->set_sliced_input(Yi, Y, 0, 1, 1, -1, 0);
    tensor_iterator->set_merged_input(M_body, M, Zo);

    // Output 0 is last Zo
    auto out1 = tensor_iterator->get_iter_value(Zo, -1);
    function = std::make_shared<ov::Model>(ov::OutputVector{out1}, ov::ParameterVector{X, Y, M});

    std::vector<float> inputX{2, 3}, inputY{4, 5}, inputM{5};
    std::vector<float> expected_result{240}; // 240 = (2+4)*(3+5)*5
    Tensor inp_tensorX(ov::element::f32, ov::Shape{1, 2}, inputX);
    Tensor inp_tensorY(ov::element::f32, ov::Shape{2, 1}, inputY);
    Tensor inp_tensorM(ov::element::f32, ov::Shape{1, 1}, inputM);
    Tensor exp_tensor(ov::element::f32, ov::Shape{1, 1}, expected_result);
    inputData = {inp_tensorX.data, inp_tensorY.data, inp_tensorM.data};
    refOutData = {exp_tensor.data};

    Exec();
}
