// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique.hpp"

#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
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
    auto const_empty = v0::Constant::create(element::i64, Shape{}, {0});

    bool sorted = true;
    bool return_inverse = false;
    bool return_counts = false;
    if (!context.input_is_none(1)) {
        sorted = context.const_input<bool>(1);
    }
    if (!context.input_is_none(2)) {
        return_inverse = context.const_input<bool>(2);
    }
    if (!context.input_is_none(3)) {
        return_counts = context.const_input<bool>(3);
    }

    OutputVector result;
    auto outputs = context.mark_node(std::make_shared<v10::Unique>(x, sorted));
    result.push_back(outputs->output(0));
    if (return_inverse) {
        result.push_back(outputs->output(2));
    } else {
        result.push_back(const_empty);
    }
    if (return_counts) {
        result.push_back(outputs->output(3));
    } else {
        result.push_back(const_empty);
    }

    return result;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
