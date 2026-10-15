// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    // One numbering for both ingest paths: every case below is reachable from the llama.cpp cgraph
    // decoder (see ggml-decoder.cpp::compute_op_case) and from the native .gguf builder, which
    // describes its reshapes so that the same case applies.
    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4 || op_case == 5 ||
                                    op_case == 6 || op_case == 7 || op_case == 8,
                                "Unsupported RESHAPE case");

    if (op_case == 8) {
        // Identity reshape (ggml src ne == node ne): a no-op. Pass the input through so any dynamic
        // token axis it carries is preserved (a static reshape would bake in the compile-time count).
        return {context.get_input(0)};
    }

    auto output_shape = context.get_output_shape().to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        // [B, 1, T, n_head*head_size] -> [B, T, n_head, head_size]: split the last dim into heads
        // and flatten whatever leads it into dim 1. The 3D form was causing RoPE broadcasting to
        // T×T when the trailing dimensions are 1 (MQA, n_head_kv=1).
        //
        // The leading dim is COPIED from the input via special_zero rather than a literal 1, so
        // the attention block stays layout-polymorphic: ov::pass::SDPAToPagedAttention moves the
        // token count into dim 0, and a literal here would discard that. With the 0 the same
        // constant serves both:
        //   SDPA inference: in [1, 1, T, H*S]  -> [1, T, H, S]
        //   PagedAttention: in [T, 1, 1, H*S]  -> [T, 1, H, S]  (identical buffer, tokens in dim 0)
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{0, -1, (int64_t)output_shape[2], (int64_t)output_shape[3]});
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, /*special_zero=*/true)},
            context.get_name());
    } else if (op_case == 2) {
        // Merge the heads back after attention. Like op_case 1, the leading dim is copied
        // (special_zero) so the token axis stays wherever the active attention backend put it.
        //
        // Rank stays 4 because the next op is the residual Add against the layer input, and OV
        // broadcasts elementwise operands from the right: every activation is rank 4 (ggml's own
        // convention), so a rank-3 result would right-align and silently form a token x token
        // outer product once the token count is not on the expected axis.
        //   in [1, T, H, S] -> [1, 1, T, H*S]
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{0, (int64_t)output_shape[1], -1, (int64_t)output_shape[3]});
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, /*special_zero=*/true)},
            context.get_name());

    } else if (op_case == 3) {
        // Flatten-for-SET_ROWS: [F, tok, 1, 1] -> [1, F*tok, -1, 1] (the KV-cache write path, e.g.
        // gpt-oss cache_v). Token count stays on the dynamic axis via -1.
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64,
            {4},
            std::vector<int64_t>{(int64_t)output_shape[0], (int64_t)output_shape[1], -1, 1});

    } else if (op_case == 4) {
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};

    } else if (op_case == 5) {
        std::vector<int64_t> shape_vec = {1, 1, -1, (int64_t)context.get_output_shape().to_shape()[3]};
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, shape_vec);

    } else if (op_case == 6) {
        // The output layout rearranges dims relative to the input (e.g. qwen3-next q/k_conv_predelta:
        // [128,2,8,T] -> [128,16,T,1]). The decoder supplies the OV-order target with -1 on the dynamic
        // token axis so the stateful model reuses across token counts; fall back to the static output
        // shape when no dynamic axis was inferred.
        auto tgt = context.get_attribute<std::vector<int64_t>>("reshape_target", {});
        if (tgt.empty()) {
            tgt.assign(output_shape.begin(), output_shape.end());
        }
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt);

    } else if (op_case == 7) {
        // General fully-static reshape (no dynamic token axis): reshape straight to the static
        // output shape. Used by qwen3-next's recurrent-state predelta reshape [262144]->[16,128,128].
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64,
                                                      {output_shape.size()},
                                                      std::vector<int64_t>(output_shape.begin(), output_shape.end()));
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
