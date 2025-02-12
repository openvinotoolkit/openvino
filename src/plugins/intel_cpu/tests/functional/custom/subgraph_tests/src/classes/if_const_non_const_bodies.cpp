// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "if_const_non_const_bodies.hpp"

#include <memory>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string IfConstNonConst::getTestCaseName(const testing::TestParamInfo<IfConstNonConstTestParams>& obj) {
    bool isThenConst;
    bool isElseConst;
    std::tie(isThenConst, isElseConst) = obj.param;
    std::ostringstream result;
    result << "isThenConst=" << isThenConst << "_";
    result << "isElseConst=" << isElseConst;
    return result.str();
}

void IfConstNonConst::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;

    // @todo pass as a test argument
    std::vector<InputShape> inputShapes{std::make_pair(PartialShape{-1}, std::vector<Shape>{{1}, {10}, {1}})};
    init_input_shapes(inputShapes);
    // @todo pass as a test argument
    bool then_body_is_constant = std::getenv("THEN_CONST") ? true : false;
    bool else_body_is_constant = std::getenv("ELSE_CONST") ? true : false;

    // Model = Parameter -> Add -> Result
    auto create_if_body = [](const ov::PartialShape& input_shape,
                             bool is_constant) -> std::tuple<std::shared_ptr<ov::op::v0::Parameter>,
                                                             std::shared_ptr<ov::op::v0::Result>,
                                                             std::shared_ptr<ov::Model>> {
        ov::Shape add_const_shape{10};  // Assuming a single-element per iteration
        auto create_constant = [](const ov::Shape& shape) -> std::shared_ptr<ov::Node> {
            auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape);
            return std::make_shared<ov::op::v0::Constant>(tensor);
        };

        auto then_param = is_constant ? nullptr : std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        std::shared_ptr<ov::Node> then_input = is_constant ? create_constant(add_const_shape) : then_param;

        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, add_const_shape);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        auto add = std::make_shared<ov::op::v1::Add>(then_input, constant);
        auto then_result = std::make_shared<ov::op::v0::Result>(add);
        auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_result});

        return {then_param, then_result, then_body};
    };

    ov::PartialShape input_shape = inputDynamicShapes.front();
    auto [then_param, then_result, then_body] = create_if_body(input_shape, then_body_is_constant);
    auto [else_param, else_result, else_body] = create_if_body(input_shape, else_body_is_constant);

    ov::Shape cond_const_shape{10};
    auto if_cond_input = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, cond_const_shape);
    auto if_value_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    // 'If' operation
    auto if_op = std::make_shared<ov::op::v8::If>(if_cond_input);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    auto then_input_param = then_body_is_constant ? nullptr : then_param;
    auto else_input_param = else_body_is_constant ? nullptr : else_param;
    if_op->set_input(if_value_input, then_input_param, else_input_param);
    if_op->set_output(then_result, else_result);
    auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

    auto ti_cond_input = std::make_shared<ov::op::v0::Constant>(
        ov::element::boolean,
        cond_const_shape,
        std::vector<bool>{true, true, false, true, false, true, true, false, false, false});
    // auto ti_cond_input = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, input_shape);
    auto ti_value_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    // TensorIterator to loop over input and connect If operation within its body
    auto ti_op = std::make_shared<ov::op::v0::TensorIterator>();
    auto ti_model =
        std::make_shared<ov::Model>(ov::ResultVector{if_result}, ov::ParameterVector{if_cond_input, if_value_input});
    ti_op->set_body(ti_model);
    // configure ti inputs mapping
    ti_op->set_sliced_input(if_cond_input, ti_cond_input, 0, 1, 1, -1, 0);
    ti_op->set_invariant_input(if_value_input, ti_value_input);
    ti_op->get_iter_value(if_result);
    auto result = std::make_shared<ov::op::v0::Result>(ti_op->output(0));

    function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ti_value_input});
}

TEST_P(IfConstNonConst, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
