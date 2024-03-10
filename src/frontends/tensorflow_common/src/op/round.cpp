// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/round.hpp"
#include<complex>
#include<vector>
#include"helper_ops/boolvariant.hpp"
#include "common_op_table.hpp"
#include<concat.hpp>
#include<tensor.hpp>
#include <host_tensor.hpp>
#include"helper_ops/complex_tensor_reshape.hpp"
#include "helper_ops/flatten_tensor.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

std::vector<size_t> get_shape(const ov::Output<ov::Node>& tensor) {
    auto host_tensor_ptr = tensor.get_tensor_ptr();
    auto shape = host_tensor_ptr->get_shape();
    
    // Convert the shape to a std::vector<size_t> and return it.
    std::vector<size_t> shape_vector(shape.begin(), shape.end());
    return shape_vector;
}

OutputVector reshape_complex_tensor(const ov::Output<ov::Node>& real, const ov::Output<ov::Node>& imag, const ov::Output<ov::Node>& shape) {
    // Concatenate the shape with [2] to account for the real and imaginary parts.
    OutputVector concat_inputs;
    concat_inputs.push_back(shape);
    concat_inputs.push_back(make_shared<v0::Constant>(shape.get_element_type(), Shape{1}, 2));
    auto concat = make_shared<v0::Concat>(concat_inputs, 0);

    // Reshape the real and imaginary parts.
    auto real_reshape = make_shared<v1::Reshape>(real, concat, false);
    auto imag_reshape = make_shared<v1::Reshape>(imag, concat, false);

    // Wrap the reshaped real and imaginary parts in a ComplexTensor.
    ComplexTensor complexTensor(real_reshape, imag_reshape);

    // Wrap the ComplexTensor in an ov::Node and return it.
    auto complex_node = make_shared<ov::Node>(complexTensor);
    return {complex_node};
}

OutputVector translate_round_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Round", "ROUND"});

    auto input = node.get_input(0);
    if (input.get_rt_info().count("ComplexTypeMark")) {
        // Propagate the ComplexTypeMark to the output tensor
        auto res = make_shared<v5::Round>(input, v5::Round::RoundMode::HALF_TO_EVEN);
        res->get_rt_info()["ComplexTypeMark"] = std::make_shared<ov::BoolVariant>(true);
        auto flatten_in = flatten(input);
        auto tensor_shape = get_shape(input); 

        // Extract the real and imaginary parts of the tensor
        std::vector<float> real(flatten_in.size());
        std::vector<float> imag(flatten_in.size());
        for (int i = 0; i < flatten_in.size(); ++i) {
            real[i] = flatten_in[i].real();
            imag[i] = flatten_in[i].imag();
        }

        auto real_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{real.size()}, real);
        auto imag_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{imag.size()}, imag);
        auto shape_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{tensor_shape.size()}, tensor_shape);
        auto reshapedComplexTensor = reshape_complex_tensor(real_node, imag_node, shape_node);

        // Apply round operation on reshaped complex tensor
        auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
        auto roundedComplexTensor = make_shared<v5::Round>(reshapedComplexTensor[0], round_mode);
        set_node_name(node.get_name(), roundedComplexTensor);

        return {roundedComplexTensor->output(0)};
    } else {
        // using default round mode "half_to_even" in openvino,
        // as TF has only that mode
        auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
        auto res = make_shared<v5::Round>(input, round_mode);
        set_node_name(node.get_name(), res);
        return {res->output(0)};
    }
} // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov