// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/elu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_unary_elu(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    // ggml's op_elu is `x > 0 ? x : expm1(x)`, i.e. ELU with alpha fixed at 1; it takes no param.
    auto res = std::make_shared<ov::op::v0::Elu>(input, 1.0);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
