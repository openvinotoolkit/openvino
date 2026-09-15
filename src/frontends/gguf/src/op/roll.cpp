// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_roll(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    const auto shifts = context.get_attribute<std::vector<int64_t>>("roll_shifts");
    FRONT_END_OP_CONVERSION_CHECK(shifts.size() == 4, "ROLL requires one shift per axis");

    auto shift = ov::op::v0::Constant::create(ov::element::i64, {4}, shifts);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, 2, 3});
    auto result = std::make_shared<ov::op::v7::Roll>(context.get_input(0), shift, axes);
    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
