// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_CUMSUM computes a prefix sum along ggml dim 0 (the innermost/fastest dimension).
// The frontend works in OV layout (ggml [ne0, ne1, ne2, ne3] -> OV [ne3, ne2, ne1, ne0]), so
// ggml dim 0 is the last OV axis (-1).
OutputVector translate_cumsum(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1LL});
    auto res = std::make_shared<ov::op::v0::CumSum>(x, axis);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
