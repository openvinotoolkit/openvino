// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_any_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    bool keep_dims = false;

    if (!context.input_is_none(1)) {
        auto dims = context.get_input(1);
        if (!context.input_is_none(2))
            keep_dims = context.const_input<bool>(2);
        auto any = context.mark_node(std::make_shared<ov::op::v1::ReduceLogicalOr>(x, dims, keep_dims));
        return {any};
    } else {
        auto axes = get_axes_range(context, 0);
        auto any = context.mark_node(std::make_shared<ov::op::v1::ReduceLogicalOr>(x, axes, keep_dims));
        return {any};
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
