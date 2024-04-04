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
                                      // (input tensor, dim = none, keepdim = false, out = none)

    auto input = context.get_input(0);

    // check if dim is provided, if not, get the range of axes to compute min and max
    auto dim = !context.input_is_none(1) ? context.get_input(1) : get_axes_range(context, 0);

    // check if keepdim is provided, if not, set it to false like PyTorch
    bool keep_dims = !context.input_is_none(2) ? context.const_input<bool>(2) : false;

    auto amin = context.mark_node(std::make_shared<v1::ReduceMin>(input, dim, keep_dims));
    auto amax = context.mark_node(std::make_shared<v1::ReduceMax>(input, dim, keep_dims));

    if (!context.input_is_none(3)) {
        auto concat = context.mark_node(std::make_shared<v0::Concat>(OutputVector{amin, amax}, 0));
        context.mutate_input(3, concat);
    }
    return {amin, amax};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov