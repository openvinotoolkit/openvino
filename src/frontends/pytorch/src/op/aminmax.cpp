// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_aminmax(const NodeContext& context) {
    num_inputs_check(context, 1, 4);  // Expect between 1 and 4 inputs

    auto input = context.get_input(0);
    auto dim = context.get_input_size() > 1 ? context.get_input(1) : Output<Node>();
    auto keepdim = context.get_input_size() > 2 ? static_cast<bool>(context.get_input(2).get_node()) : false;

    auto amin = context.mark_node(std::make_shared<v1::ReduceMin>(input, dim, keepdim));
    auto amax = context.mark_node(std::make_shared<v1::ReduceMax>(input, dim, keepdim));

    if (!context.input_is_none(3)) {
        auto result = context.mark_node(std::make_shared<v0::Concat>(OutputVector{amin, amax}, 0));
        context.mutate_input(3, result);
    }
    return {amin, amax};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov