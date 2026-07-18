// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/reduce_sum.hpp>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_SUM_ROWS: sum along the last (row) dimension, keeping rank.
OutputVector translate_sum_rows(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
