// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/transpose.hpp>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_transpose(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    // The permutation is a static ggml layout fact (derived from input/output strides). The decoder
    // computes it and exposes it as the typed "perm" attribute, so the translator never inspects
    // ggml strides. Fall back to the classic swap-last-two-dims order when the attribute is absent
    // (e.g. single-op tests), matching ggml's plain GGML_OP_TRANSPOSE.
    auto perm = context.get_attribute<std::vector<int64_t>>("perm", {0, 1, 3, 2});

    auto res = std::make_shared<ov::op::v1::Transpose>(
        context.get_input(0),
        ov::op::v0::Constant::create(ov::element::i64, {perm.size()}, perm));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
