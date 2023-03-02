// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_argsort(NodeContext& context) {
    const auto input_tensor = context.get_input(0);
    bool stable, descending;
    int64_t dim;

    if (context.get_input_size() == 4) {
        stable = context.const_input<bool>(1);
        dim = context.const_input<int64_t>(2);
        descending = context.const_input<bool>(3);
        FRONT_END_OP_CONVERSION_CHECK(stable == false, "Stable sorting in aten::argsort is not yet supported.");
    } else {
        dim = context.const_input<int64_t>(1);
        descending = context.const_input<bool>(2);
    }

    auto sort = sort_elements(context, input_tensor, stable, dim, descending);
    return {sort->output(1)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
