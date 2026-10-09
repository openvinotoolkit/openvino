// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_permute(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto order = context.get_attribute<std::vector<int64_t>>("perm");
    const auto input = context.get_input(0);
    const auto rank = input.get_partial_shape().rank();
    if (rank.is_static() && static_cast<size_t>(rank.get_length()) < order.size()) {
        // The decoder describes the permutation over ggml's four axes; a hidden state that has already
        // dropped its leading axes keeps only the trailing ones. The dropped axes must stay in place.
        const size_t drop = order.size() - static_cast<size_t>(rank.get_length());
        for (size_t axis = 0; axis < drop; ++axis) {
            FRONT_END_OP_CONVERSION_CHECK(order[axis] == static_cast<int64_t>(axis),
                                          "PERMUTE moves an axis that the operand no longer has: operand rank ",
                                          rank.get_length(),
                                          ", permutation covers ",
                                          order.size(),
                                          " axes");
        }
        order.erase(order.begin(), order.begin() + drop);
        for (auto& axis : order) {
            axis -= static_cast<int64_t>(drop);
        }
    }
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {order.size()}, order);
    return rename_outputs_with_suffix({std::make_shared<ov::op::v1::Transpose>(input, perm)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
