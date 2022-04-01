// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_simple.hpp"
#include "common_test_utils/data_utils.hpp"
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
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto add = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{data0, data1},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Add>(indata0, indata1)},
                                                                      ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> AddSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sin0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sin1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto add = std::make_shared<op::v1::Add>(sin0, sin1);
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> AddSinhFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sin0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sin1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, sin0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, sin1->get_shape());
    auto add = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{data0, data1},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Add>(sin0, sin1)},
                                                                      ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(input_shapes[1]), -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, data1->get_shape(), const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(shape_size(input_shapes[1]), -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, data1->get_shape(), const_values);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto indata2 = std::make_shared<op::v0::Parameter>(precision, data1->get_shape());
    auto add = std::make_shared<op::v1::Add>(indata0, indata1);
    auto sub = std::make_shared<op::v1::Subtract>(add, const_data);
    auto mul = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{data0, data1, const_data},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Multiply>(add, sub)},
                                                                  ParameterVector{indata0, indata1, indata2}));
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> EltwiseThreeInputsFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(1, -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(precision, Shape{1}, const_values);
    auto add = std::make_shared<op::v1::Add>(data0, data1);
    auto sub = std::make_shared<op::v1::Subtract>(data2, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> EltwiseThreeInputsSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto sinh0 = std::make_shared<op::v0::Sinh>(data0);
    auto sinh1 = std::make_shared<op::v0::Sinh>(data1);
    auto sinh2 = std::make_shared<op::v0::Sinh>(data2);
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(1, -10., 10.);
    auto const_data = std::make_shared<op::v0::Constant>(ov::element::f32, Shape{1}, const_values);
    auto add = std::make_shared<op::v1::Add>(sinh0, sinh1);
    auto sub = std::make_shared<op::v1::Subtract>(sinh2, const_data);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub);
    return std::make_shared<ov::Model>(NodeVector{mul}, ParameterVector{data0, data1, data2});
}

std::shared_ptr<ov::Model> MatMulEltwiseBranchesFunction::initOriginal() const {
    auto data_1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data_2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(4, -10., 10.);
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
    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(4, -10., 10.);
    // snippet inputs
    auto non_snippet_op = std::make_shared<op::v0::MatMul>(data_1, data_2);
    auto mul_const_1 = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[0]);
    auto add_const_1 = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[1]);
    auto mul_const_2 = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[2]);
    auto sub_const_2 = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[3]);

    // snippet function
    Shape matMulOutShape = input_shapes[0];
    matMulOutShape.back() = input_shapes[1].back();
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

    ngraph::NodeVector snippet_inputs{ non_snippet_op };
    auto snippet = std::make_shared<ngraph::snippets::op::Subgraph>(snippet_inputs, snippet_function);
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
    auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{data0, data1}, body);
    auto log = std::make_shared<op::v0::Log>(subgraph->output(0));
    //Note that log is not currently supported by snippets, so it won't be converted to subgraph.
    // Todo: Note that collapse_subgraph changes the output ports so that the input subgraph's outputs come
    //  before the node outputs. So the Subgraph{Add}.output(1)->Log{} becomes Subgraph{Add+Hswish}.output(0)->Log{}
    auto subgraph_param = std::make_shared<op::v0::Parameter>(precision, subgraph->get_output_shape(1));
    auto log_param = std::make_shared<op::v0::Parameter>(precision, log->get_output_shape(0));
    auto mul = std::make_shared<ngraph::snippets::op::Subgraph>(OutputVector{subgraph->output(1), log->output(0)},
                                          std::make_shared<Model>(NodeVector{std::make_shared<op::v1::Multiply>(subgraph_param, log_param)},
                                                                  ParameterVector{subgraph_param, log_param}));
    return std::make_shared<Model>(NodeVector{mul}, ParameterVector{data0, data1});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov