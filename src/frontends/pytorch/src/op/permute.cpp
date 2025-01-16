// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto data = context.get_input(0);
    auto order = get_input_concat_if_list(context, 1);
    auto rank = std::get<1>(get_shape_rank(context, data));
    auto rank_converted = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(rank, order));
    auto order_normalized = normalize_axis(context, order, rank_converted);
    if (const auto order_const = ov::util::get_constant_from_source(order_normalized)) {
        order_normalized = order_const;
    }
    return {context.mark_node(std::make_shared<ov::op::v1::Transpose>(data, order_normalized))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
