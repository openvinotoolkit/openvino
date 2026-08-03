// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/op/multiply.hpp"
#include "openvino/op/sqrt.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_SQR: element-wise square (x * x).
OutputVector translate_sqr(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, input);

    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_SQRT: element-wise square root.
OutputVector translate_sqrt(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto res = std::make_shared<ov::op::v0::Sqrt>(context.get_input(0));

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
