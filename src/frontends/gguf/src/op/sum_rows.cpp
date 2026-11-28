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

// GGML_OP_SUM_ROWS: sum along the last (row) dimension, keeping rank. Native builders may select
// other OpenVINO-order "axes" and drop them with "keep_dims" = false.
OutputVector translate_sum_rows(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    const auto axes = context.get_attribute<std::vector<int64_t>>("axes", {-1});
    auto res = std::make_shared<ov::op::v1::ReduceSum>(
        context.get_input(0),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes),
        context.get_attribute<bool>("keep_dims", true));

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
