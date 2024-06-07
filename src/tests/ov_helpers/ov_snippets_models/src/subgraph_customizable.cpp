// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_customizable.hpp"
#include "openvino/op/util/op_types.hpp"
#include <snippets/op/subgraph.hpp>
#include "common_test_utils/data_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> ConvMulActivationFunction::initOriginal() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto channels = static_cast<size_t>(input_shapes[0][1].get_length());
    ov::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const Shape const_shape {channels, channels, 3, 3};
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto eltwise_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto eltwise_sinh = std::make_shared<op::v0::Sinh>(eltwise_param);

    auto eltwise_binary = custom_ops[0]->clone_with_new_inputs({conv->output(0), eltwise_sinh->output(0)});
    auto eltwise_unary_1 = custom_ops[1]->clone_with_new_inputs({eltwise_binary->output(0)});
    auto eltwise_unary_2 = custom_ops[2]->clone_with_new_inputs({eltwise_unary_1->output(0)});

    return std::make_shared<ov::Model>(NodeVector{eltwise_unary_2}, ParameterVector{conv_param, eltwise_param});
}
std::shared_ptr<ov::Model> ConvMulActivationFunction::initReference() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ov::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const auto channels = static_cast<size_t>(input_shapes[0][1].get_length());
    const Shape const_shape {channels, channels, 3, 3};
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    auto eltwise_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto eltwise_sinh = std::make_shared<op::v0::Sinh>(eltwise_param);

    auto indata0 = std::make_shared<op::v0::Parameter>(precision, conv->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, eltwise_sinh->get_shape());

    auto ineltwise_binary = custom_ops[0]->clone_with_new_inputs({indata0->output(0), indata1->output(0)});
    auto ineltwise_unary_1 = custom_ops[1]->clone_with_new_inputs({ineltwise_binary->output(0)});
    auto ineltwise_unary_2 = custom_ops[2]->clone_with_new_inputs({ineltwise_unary_1->output(0)});

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{conv, eltwise_sinh},
                                          std::make_shared<ov::Model>(NodeVector{ineltwise_unary_2},
                                                                  ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{conv_param, eltwise_param});
}
std::shared_ptr<ov::Model> ConvBiasActivationFunction::initOriginal() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto channels = static_cast<size_t>(input_shapes[0][1].get_length());
    ov::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const Shape const_shape {channels, channels, 3, 3};
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const Shape add_const_shape = {batches, channels, 1, 1};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);

    auto add = std::make_shared<op::v1::Add>(conv->output(0), add_const->output(0));
    auto unary = custom_ops[1]->clone_with_new_inputs({add->output(0)});

    return std::make_shared<ov::Model>(NodeVector{unary}, ParameterVector{conv_param});
}
std::shared_ptr<ov::Model> ConvBiasTwoActivationFunction::initOriginal() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto channels = static_cast<size_t>(input_shapes[0][1].get_length());
    ov::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const Shape const_shape {channels, channels, 3, 3};
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const Shape add_const_shape = {batches, channels, 1, 1};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);

    auto add = std::make_shared<op::v1::Add>(conv->output(0), add_const->output(0));
    auto unary_1 = custom_ops[1]->clone_with_new_inputs({add->output(0)});
    auto unary_2 = custom_ops[2]->clone_with_new_inputs({unary_1->output(0)});

    return std::make_shared<ov::Model>(NodeVector{unary_2}, ParameterVector{conv_param});
}
std::shared_ptr<ov::Model> ConvBiasTwoActivationFunction::initReference() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    const auto channels = static_cast<size_t>(input_shapes[0][1].get_length());
    ov::Shape strides(2, 1);
    std::vector<ptrdiff_t> pad_begin(2, 1), pad_end(2, 1);
    const Shape const_shape {channels, channels, 3, 3};
    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(shape_size(const_shape), -10., 10.);
    auto weights = std::make_shared<op::v0::Constant>(precision, const_shape, const_values);
    auto conv = std::make_shared<op::v1::Convolution>(conv_param, weights, strides, pad_begin, pad_end, strides);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const Shape add_const_shape = {batches, channels, 1, 1};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);

    auto add = std::make_shared<op::v1::Add>(conv->output(0), add_const->output(0));
    auto unary_1 = custom_ops[1]->clone_with_new_inputs({add->output(0)});

    auto indata = std::make_shared<op::v0::Parameter>(precision, unary_1->get_shape());

    auto unary_2 = custom_ops[2]->clone_with_new_inputs({indata->output(0)});

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{unary_1},
                                                                 std::make_shared<ov::Model>(NodeVector{unary_2},
                                                                 ParameterVector{indata}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{conv_param});
}
std::shared_ptr<ov::Model> MatMulTwoActivationFunction::initOriginal() const {
    auto matmul_param0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto matmul_param1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto matmul = std::make_shared<op::v0::MatMul>(matmul_param0, matmul_param1);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const auto channels = static_cast<size_t>(input_shapes[1][1].get_length());
    const Shape add_const_shape = {batches, 1, 1, channels};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);

    auto add = std::make_shared<op::v1::Add>(matmul, add_const);
    auto unary_1 = custom_ops[1]->clone_with_new_inputs({add->output(0)});
    auto unary_2 = custom_ops[2]->clone_with_new_inputs({unary_1->output(0)});

    return std::make_shared<ov::Model>(NodeVector{unary_2}, ParameterVector{matmul_param0, matmul_param1});
}
std::shared_ptr<ov::Model> MatMulBiasActivationBinaryFunction::initOriginal() const {
    auto matmul_param0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto matmul_param1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto matmul = std::make_shared<op::v0::MatMul>(matmul_param0, matmul_param1);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const auto channels = static_cast<size_t>(input_shapes[1][1].get_length());
    const Shape add_const_shape = {batches, 1, 1, channels};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);
    auto add = std::make_shared<op::v1::Add>(matmul, add_const);

    auto unary = custom_ops[1]->clone_with_new_inputs({add->output(0)});

    auto binary_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto binary = custom_ops[2]->clone_with_new_inputs({unary->output(0), binary_param});

    return std::make_shared<ov::Model>(NodeVector{binary}, ParameterVector{matmul_param0, matmul_param1, binary_param});
}
std::shared_ptr<ov::Model> MatMulBiasActivationBinaryFunction::initReference() const {
    auto matmul_param0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto matmul_param1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto matmul = std::make_shared<op::v0::MatMul>(matmul_param0, matmul_param1);

    const auto batches = static_cast<size_t>(input_shapes[0][0].get_length());
    const auto channels = static_cast<size_t>(input_shapes[1][1].get_length());
    const Shape add_const_shape = {batches, 1, 1, channels};
    const std::vector<float> add_const_values = ov::test::utils::generate_float_numbers(shape_size(add_const_shape), -10., 10.);
    auto add_const = std::make_shared<op::v0::Constant>(precision, add_const_shape, add_const_values);
    auto add = std::make_shared<op::v1::Add>(matmul, add_const);

    auto unary = custom_ops[1]->clone_with_new_inputs({add->output(0)});

    auto binary_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);

    auto indata0 = std::make_shared<op::v0::Parameter>(precision, unary->get_shape());
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, binary_param->get_shape());

    auto binary = custom_ops[2]->clone_with_new_inputs({indata0->output(0), indata1->output(0)});

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{unary, binary_param},
                                                                 std::make_shared<ov::Model>(NodeVector{binary},
                                                                 ParameterVector{indata0, indata1}));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{matmul_param0, matmul_param1, binary_param});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov