// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include <memory>

#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unique.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_unique2(const NodeContext& context) {
    // torch.unique(input, sorted=True, return_inverse=False, return_counts=False) →
    // Tuple[Tensor, Tensor, Tensor]
    num_inputs_check(context, 1, 4);
    auto x = context.get_input(0);
    auto const_empty = std::make_shared<v0::Constant>(element::i64, Shape{0}, std::vector<int64_t>{});
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));

    bool return_inverse = false;
    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_inverse = context.const_input<bool>(2);
    }
    if (!context.input_is_none(3)) {
        return_counts = context.const_input<bool>(3);
    }

    OutputVector result;
    // Since PyTorch 2.2.0 the `Unique` op always returns sorted values, regardless of the parameter `sorted`.
    // Reference: pytorch/pytorch#105742, pytorch/pytorch#113420,
    // https://pytorch.org/docs/1.13/generated/torch.unique.html#torch.unique
    auto outputs = context.mark_node(std::make_shared<v10::Unique>(x, true));
    auto unique_values = outputs->output(0);
    result.push_back(unique_values);
    if (return_inverse) {
        auto x_shape = context.mark_node(std::make_shared<v0::ShapeOf>(x));
        auto inverse = context.mark_node(std::make_shared<v1::Reshape>(outputs->output(2), x_shape, false));
        result.push_back(inverse);
    } else {
        result.push_back(const_empty);
    }
    if (return_counts) {
        auto counts = outputs->output(3);
        result.push_back(counts);
    } else {
        result.push_back(const_empty);
    }

    return result;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
