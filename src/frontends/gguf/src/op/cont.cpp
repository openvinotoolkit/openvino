// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <memory>

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_attribute<int>("op_case", 0);
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");

    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE. translate_permute already emitted a real ov::Transpose, so
        // the OV tensor is logically contiguous in the permuted layout -- CONT (which only makes the
        // ggml memory contiguous) is a no-op for us. gemma3n/gemma4 use CONT(PERMUTE(inp_per_layer))
        // before slicing per-layer embeddings; keeping this in-OV avoids a host round-trip.
        return {context.get_input(0)};
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};
    } else {
        // The input comes from a VIEW
        res = process_view_input(context, 0);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
