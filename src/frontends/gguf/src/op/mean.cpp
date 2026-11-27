// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_MEAN: mean along the last (row) dimension, keeping rank.
OutputVector translate_mean(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto res =
        std::make_shared<ov::op::v1::ReduceMean>(context.get_input(0),
                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                                                 true);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
