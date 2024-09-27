// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_simple.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> AddFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> AddFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1}, getOriginal());
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> ExpFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto exp = std::make_shared<op::v0::Exp>(data0);
    return std::make_shared<ov::Model>(NodeVector{exp}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ExpReciprocalFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto factor = std::make_shared<op::v0::Constant>(precision, ov::Shape{1}, std::vector<float>{-1.f});
    auto exp = std::make_shared<op::v0::Exp>(data0);
    auto reciprocal = std::make_shared<op::v1::Power>(exp, factor);
    return std::make_shared<ov::Model>(NodeVector{reciprocal}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> AddConstFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(m_const_shape.get_shape()), -10., 10.);
    auto const_data1 = std::make_shared<op::v0::Constant>(precision, m_const_shape.get_shape(), const_values);
    auto add = std::make_shared<op::v1::Add>(data0, const_data1);
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> AddRollConstFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(m_const_shape.get_shape()), -10., 10.);
    auto const_data1 = std::make_shared<op::v0::Constant>(precision, m_const_shape.get_shape(), const_values);
    auto shift = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<float>{1});
    auto axes = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<float>{0});
    auto roll0 = std::make_shared<ov::op::v7::Roll>(data0, shift, axes);
    auto add = std::make_shared<op::v1::Add>(roll0, const_data1);
    // The limitation for BF16 in CPU Plugin:
    roll0->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    add->get_rt_info()["enforceBF16evenForGraphTail"] = true;
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> EltwiseFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(1, -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, data1->get_shape(), const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(1, -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, data1->get_shape(), const_values);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto indata2 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto sub = std::make_shared<op::v1::Subtract>(add, indata2);
    auto mul = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1, const_data},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Multiply>(add, sub)},
                                                                  ParameterVector{indata0, indata1, indata2}));
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> EltwiseThreeInputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(1, -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(data2, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> EltwiseMaxNumParamsFunction::initOriginal() const {
    ParameterVector params;
    for (const auto& shape : input_shapes) {
        auto param = std::make_shared<op::v0::Parameter>(precision, shape);
        params.push_back(param);
    }
    std::vector<std::shared_ptr<Node>> add; // 4
    for (size_t i = 0; i < input_shapes.size() / 2; i++) {
        add.push_back(std::make_shared<op::v1::Add>(params[i * 2], params[i * 2 + 1]));
    }
    std::vector<std::shared_ptr<Node>> mul; // 2
    for (size_t i = 0; i < add.size() / 2; i++) {
        auto mul_node = std::make_shared<op::v1::Multiply>(add[i * 2], add[i * 2 + 1]);
        mul.push_back(mul_node);
    }
    auto sub = std::make_shared<op::v1::Subtract>(mul[0], mul[1]);
    auto power = std::make_shared<op::v1::Power>(params.back(), sub);
    auto exit_sinh = std::make_shared<op::v0::Sinh>(power);
    return std::make_shared<ov::Model>(NodeVector{sub, exit_sinh}, params);
}

std::shared_ptr<ov::Model> MatMulEltwiseBranchesFunction::initOriginal() const {
    auto data_1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data_2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh_1 = std::make_shared<op::v0::Sinh>(data_1);
    auto sinh_2 = std::make_shared<op::v0::Sinh>(data_2);
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(sinh_1, sinh_2);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(4, -10., 10.);
    auto mul_const_1 = op::v0::Constant::create(precision, {1}, {const_values[0]});
    auto mul_1 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_1);
    auto add_const_1 = op::v0::Constant::create(precision, {1}, {const_values[1]});
    auto add_1 = std::make_shared<op::v1::Add>(mul_1, add_const_1);
    auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

    auto mul_const_2 = op::v0::Constant::create(precision, {1}, {const_values[2]});
    auto mul_2 = std::make_shared<op::v1::Multiply>(non_snippet_op, mul_const_2);
    auto sub_const_2 = op::v0::Constant::create(precision, {1}, {const_values[3]});
    auto sub_2 = std::make_shared<op::v1::Subtract>(mul_2, sub_const_2);
    auto relu = std::make_shared<op::v0::Relu>(sub_2);

    auto add = std::make_shared<op::v1::Add>(elu, relu);
    auto result = std::make_shared<op::v0::Result>(add);

    return std::make_shared<Model>(ResultVector{ result }, ParameterVector{ data_1, data_2 });
}

std::shared_ptr<ov::Model> MatMulEltwiseBranchesFunction::initReference() const {
    auto data_1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data_2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh_1 = std::make_shared<op::v0::Sinh>(data_1);
    auto sinh_2 = std::make_shared<op::v0::Sinh>(data_2);
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(4, -10., 10.);
    // snippet inputs
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(sinh_1, sinh_2);
    auto mul_const_1 = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values[0]);
    auto add_const_1 = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values[1]);
    auto mul_const_2 = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values[2]);
    auto sub_const_2 = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values[3]);

    // snippet function
    Shape matMulOutShape = input_shapes[0].get_shape();
    matMulOutShape.back() = input_shapes[1].get_shape().back();
    auto snippet_input = std::make_shared<op::v0::Parameter>(precision, matMulOutShape);

    auto mul_1 = std::make_shared<op::v1::Multiply>(snippet_input, mul_const_1);
    auto add_1 = std::make_shared<op::v1::Add>(mul_1, add_const_1);
    auto elu = std::make_shared<op::v0::Elu>(add_1, 0.01);

    auto mul_2 = std::make_shared<op::v1::Multiply>(snippet_input, mul_const_2);
    auto sub_2 = std::make_shared<op::v1::Subtract>(mul_2, sub_const_2);
    auto relu = std::make_shared<op::v0::Relu>(sub_2);

    auto add = std::make_shared<op::v1::Add>(elu, relu);
    ParameterVector subgraph_params{ snippet_input };
    auto snippet_function = std::make_shared<Model>(NodeVector{ add }, subgraph_params);

    ov::NodeVector snippet_inputs{ non_snippet_op };
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(snippet_inputs, snippet_function);
    auto result = std::make_shared<op::v0::Result>(snippet);

    return std::make_shared<Model>(NodeVector{ result }, ParameterVector{ data_1, data_2 });
}

std::shared_ptr<ov::Model> EltwiseLogLoopFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto hswish = std::make_shared<op::v4::HSwish>(add);
    auto log = std::make_shared<op::v0::Log>(add);
    auto mul = std::make_shared<op::v1::Multiply>(hswish, log);
    return std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseLogLoopFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto inAdd = std::make_shared<op::v1::Add>(indata0, indata1);
    auto inHswish = std::make_shared<op::v4::HSwish>(inAdd);
    auto body = std::make_shared<Model>(NodeVector{inAdd, inHswish}, ParameterVector{indata0, indata1});
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1}, body);
    auto log = std::make_shared<op::v0::Log>(subgraph->output(0));
    //Note that log is not currently supported by snippets, so it won't be converted to subgraph.
    // Todo: Note that collapse_subgraph changes the output ports so that the input subgraph's outputs come
    //  before the node outputs. So the Subgraph{Add}.output(1)->Log{} becomes Subgraph{Add+Hswish}.output(0)->Log{}
    auto subgraph_param = std::make_shared<op::v0::Parameter>(precision, subgraph->get_output_shape(1));
    auto log_param = std::make_shared<op::v0::Parameter>(precision, log->get_output_shape(0));
    auto mul = std::make_shared<ov::snippets::op::Subgraph>(OutputVector{subgraph->output(1), log->output(0)},
                                          std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Multiply>(subgraph_param, log_param)},
                                                                  ParameterVector{subgraph_param, log_param}));
    return std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> EltwiseTwoResultsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    data0->set_friendly_name("data0");
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    data1->set_friendly_name("data1");
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    add->set_friendly_name("add");
    auto hswish = std::make_shared<op::v4::HSwish>(add);
    hswish->set_friendly_name("hswish");
    auto relu = std::make_shared<op::v0::Relu>(hswish);
    relu->set_friendly_name("relu");

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto& out_tensor0 = add->get_output_tensor(0);
    ov::descriptor::set_ov_tensor_legacy_name(out_tensor0, "add_out");
    out_tensor0.set_names({"add_out", "y0"});

    auto& out_tensor1 = relu->get_output_tensor(0);
    ov::descriptor::set_ov_tensor_legacy_name(out_tensor1, "relu_out");
    out_tensor1.set_names({"relu_out", "y1"});
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto res0 = std::make_shared<op::v0::Result>(add);
    res0->set_friendly_name("res0");
    auto res1 = std::make_shared<op::v0::Result>(relu);
    res1->set_friendly_name("res1");
    return std::make_shared<Model>(ResultVector{res0, res1}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseTwoResultsFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    data0->set_friendly_name("data0");
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    data1->set_friendly_name("data1");

    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    add->set_friendly_name("add");
    auto hswish = std::make_shared<op::v4::HSwish>(add);
    hswish->set_friendly_name("hswish");
    auto subgraph0 = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1},
                                        std::make_shared<ov::Model>(NodeVector{add, hswish},
                                                                    ParameterVector{indata0, indata1}));
    subgraph0->set_friendly_name("add");
    auto indata2 = std::make_shared<op::v0::Parameter>(precision, subgraph0->get_output_shape(1));
    auto relu = std::make_shared<op::v0::Relu>(indata2);
    relu->set_friendly_name("relu");
    auto subgraph1 = std::make_shared<ov::snippets::op::Subgraph>(OutputVector{subgraph0->output(1)},
                                        std::make_shared<ov::Model>(NodeVector{relu},
                                                                    ParameterVector{indata2}));
    subgraph1->set_friendly_name("relu");
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto& out_tensor0 = subgraph0->get_output_tensor(0);
    ov::descriptor::set_ov_tensor_legacy_name(out_tensor0, "add_out");
    out_tensor0.set_names({"add_out", "y0"});

    auto& out_tensor1 = subgraph1->get_output_tensor(0);
    ov::descriptor::set_ov_tensor_legacy_name(out_tensor1, "relu_out");
    out_tensor1.set_names({"relu_out", "y1"});
    OPENVINO_SUPPRESS_DEPRECATED_END

    auto res0 = std::make_shared<op::v0::Result>(subgraph0->output(0));
    res0->set_friendly_name("res0");
    auto res1 = std::make_shared<op::v0::Result>(subgraph1->output(0));
    res1->set_friendly_name("res1");
    return std::make_shared<Model>(ResultVector{res0, res1}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TwoInputsAndOutputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto hswish = std::make_shared<op::v4::HSwish>(data0);
    auto add = std::make_shared<op::v1::Add>(hswish, data1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sin3 = std::make_shared<op::v0::Sin>(relu);

    return std::make_shared<Model>(NodeVector{hswish, sin3}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TwoInputsAndOutputsWithReversedOutputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto hswish = std::make_shared<op::v4::HSwish>(data0);
    auto add = std::make_shared<op::v1::Add>(hswish, data1);
    auto relu = std::make_shared<op::v0::Relu>(add);
    auto sin3 = std::make_shared<op::v0::Sin>(relu);

    return std::make_shared<Model>(NodeVector{sin3, hswish}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> SelectFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(ov::element::boolean, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto select = std::make_shared<op::v1::Select>(data0, data1, data2);

    return std::make_shared<Model>(NodeVector{select}, ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> BroadcastAddFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto target_shape = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{m_target_shape.size()}, m_target_shape.get_shape());
    auto broadcast = std::make_shared<ov::op::v1::Broadcast>(data0, target_shape);
    auto add = std::make_shared<op::v1::Add>(broadcast, data1);

    return std::make_shared<Model>(NodeVector{add}, ParameterVector{data0, data1});
}


std::shared_ptr<ov::Model> BroadcastSelectFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(ov::element::boolean, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto target_shape = std::make_shared<op::v0::Constant>(ov::element::i32, ov::Shape{m_target_shape.size()}, m_target_shape.get_shape());
    auto broadcast = std::make_shared<ov::op::v1::Broadcast>(data0, target_shape);
    auto select = std::make_shared<op::v1::Select>(broadcast, data1, data2);

    return std::make_shared<Model>(NodeVector{select}, ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> EdgeReplaceFunction::initOriginal() const {
    auto input = std::make_shared<op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    const auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
    // first parent and second parent have common inputs from output of split
    auto split = std::make_shared<ov::op::v1::Split>(input, axis, 2);
    auto mul_lhs = split->output(0);
    auto mul_rhs = split->output(1);

    // first parent subgraph in tokenization stage
    const auto data0 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({1, 1, 1, 3, 2}),
        std::vector<float>{0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
    const auto const_a4 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2}), std::vector<float>{0.2, 0.1});
    auto mul_a4 = std::make_shared<op::v1::Multiply>(mul_rhs, const_a4);
    auto add_a1 = std::make_shared<op::v1::Add>(mul_a4, data0);
    auto mul_a5 = std::make_shared<op::v1::Multiply>(add_a1, mul_lhs);

    // second parent subgraph in tokenization stage
    const auto data1 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({1, 1, 1, 3, 2}),
        std::vector<float>{0.1, 0.3, 0.5, 0.7, 0.9, 1.0});
    auto mul_a1 = std::make_shared<op::v1::Multiply>(data1, mul_lhs);
    const auto const_a2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2}), std::vector<float>{0.2, 0.4});
    auto mul_a2 = std::make_shared<op::v1::Multiply>(mul_a1, const_a2);
    const auto const_a3 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2}), std::vector<float>{0.3, 0.1});
    auto mul_a3 = std::make_shared<op::v1::Multiply>(mul_a2, const_a3);

    auto add_a3 = std::make_shared<op::v1::Add>(mul_a5, mul_a3);
    auto add_a2 = std::make_shared<op::v1::Add>(mul_a5, mul_a2);

    auto concat = std::make_shared<op::v0::Concat>(ov::OutputVector{add_a3, add_a2}, 0);

    return std::make_shared<Model>(NodeVector{concat}, ParameterVector{input});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
