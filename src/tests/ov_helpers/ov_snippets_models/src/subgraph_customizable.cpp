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
    ngraph::Shape strides(2, 1);
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
    eltwise_binary->set_arguments({conv->output(0), eltwise_sinh->output(0)});
    eltwise_unary_1->set_arguments({eltwise_binary->output(0)});
    eltwise_unary_2->set_arguments({eltwise_unary_1->output(0)});

    return std::make_shared<ov::Model>(NodeVector{eltwise_unary_2}, ParameterVector{conv_param, eltwise_param});
}
std::shared_ptr<ov::Model> ConvMulActivationFunction::initReference() const {
    auto conv_param = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ngraph::Shape strides(2, 1);
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
}  // namespace snippets
}  // namespace test
}  // namespace ov