// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/scatter.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    // aten::bincount(Tensor input, Tensor? weights=None, int minlength=0) -> Tensor
    num_inputs_check(context, 3, 3);

    auto input = context.get_input(0); // Input tensor (1D integers)
    auto weights = context.get_input(1); // Optional weights
    auto minlength = context.get_input(2); // Minimum output length

    // Convert input to INT32
    auto input_int = context.mark_node(std::make_shared<v0::Convert>(input, element::i32));

    // Determine output size: max(input) + 1 or minlength, whichever is larger
    auto max_val = context.mark_node(std::make_shared<v3::ReduceMax>(input_int, ov::AxisSet{0}, true));
    auto max_length = context.mark_node(std::make_shared<v1::Add>(max_val, context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}))));
    auto output_size = context.mark_node(std::make_shared<v1::Maximum>(max_length, minlength));

    // Create initial output tensor (zeros of size `output_size`)
    auto output = context.mark_node(v0::Constant::create(element::f32, Shape{output_size}, {0}));

    // Handle weights: if None, use ones
    auto weight_tensor = weights.get_node_shared_ptr() != nullptr
                         ? weights
                         : context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));

    // Scatter operation to calculate bincount
    auto result = context.mark_node(std::make_shared<v3::ScatterAdd>(output, input_int, weight_tensor));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
