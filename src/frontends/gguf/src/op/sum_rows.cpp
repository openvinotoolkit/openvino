// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_SUM_ROWS: sum along the last (row) dimension, keeping rank.
OutputVector translate_sum_rows(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // ggml sums the rows (last axis) and keeps it; builders may select another axis or drop it.
    auto input = context.get_input(0);
    const auto axis = context.get_attribute<int64_t>("axis", -1);
    auto res =
        std::make_shared<ov::op::v1::ReduceSum>(input,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis}),
                                                context.get_attribute<bool>("keep_dims", true));

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
