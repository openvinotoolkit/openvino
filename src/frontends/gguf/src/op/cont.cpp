// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <cstdint>
#include <memory>
#include <vector>
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

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
        // Input from PERMUTE: translate_permute already emitted a real Transpose, so the OV tensor
        // is logically contiguous -- CONT (a ggml memory-contiguity op) is a no-op for us.
        return {context.get_input(0)};
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};
    } else {
        // The input comes from a VIEW. Resolve the view, then reshape to the CONT's own output layout
        // (ggml_cont can merge/split dims, e.g. qwen3-next's a/b: [tok,8,2,1] -> [8*tok,2,1,1]). The
        // decoder supplies the reshape target as "cont_reshape" -- the output dims (OV order) with -1 on
        // the inferred dynamic (token) axis -- so the token count stays dynamic and a 0-token decode step
        // reshapes correctly. When the output is fully static "cont_reshape" carries no -1 (plain reshape).
        // The CONT's input is an already-resolved VIEW node (translate_view emitted its Slice/Reshape),
        // so the view offset/window is already applied -- do NOT re-slice via process_view_input, which
        // would apply the offset a second time (qwen3-next a/b: offset 2 slices axis3 [2,4) -> clamped
        // to length 0). Just reshape the resolved input to the CONT's own layout.
        auto input = context.get_input(0);
        auto tgt = context.get_attribute<std::vector<int64_t>>("cont_reshape", {});
        if (!tgt.empty()) {
            auto tgt_node = ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt);
            res = std::make_shared<ov::op::v1::Reshape>(input, tgt_node, false);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
        res = input;
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
