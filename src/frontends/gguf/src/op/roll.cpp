// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/roll.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// ggml_roll: circular shift by "roll_shifts" {shift0, shift1, shift2, shift3}, in ggml dimension
// order (dim 0 innermost). OV axis i is ggml dimension rank - 1 - i.
OutputVector translate_roll(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto shifts = context.get_attribute<std::vector<int32_t>>("roll_shifts");
    auto input = context.get_input(0);
    const auto rank = input.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static() && shifts.size() >= static_cast<size_t>(rank.get_length()),
                                  "ROLL requires a static rank and one shift per axis");
    const auto n = static_cast<size_t>(rank.get_length());
    std::vector<int64_t> shift(n);
    std::vector<int64_t> axes(n);
    for (size_t axis = 0; axis < n; ++axis) {
        shift[axis] = shifts[n - 1 - axis];
        axes[axis] = static_cast<int64_t>(axis);
    }
    auto res = std::make_shared<ov::op::v7::Roll>(input,
                                                  ov::op::v0::Constant::create(ov::element::i64, {n}, shift),
                                                  ov::op::v0::Constant::create(ov::element::i64, {n}, axes));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
