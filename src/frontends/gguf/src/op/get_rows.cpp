// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_get_rows(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    Output<Node> res;
    auto data = context.get_input(0);
    auto indices = context.get_input(1);

    if (op_case == 3) {
        return {data};
    }

    if (op_case == 4) {
        auto flat_indices =
            std::make_shared<ov::op::v0::Squeeze>(indices,
                                                  ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {2});
        res = std::make_shared<ov::op::v8::Gather>(data, flat_indices, axis);
        if (res.get_element_type() != context.get_output_type()) {
            res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
        }
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    // MoE gating-weight gather (op_case 10): data = probs [1,1,T,E], indices = selected experts
    // [1,1,T,K]; per-row (GatherElements) gather over the expert axis picks each token's K
    // selected-expert probs -> [1,1,T,K], distinct from the embedding-style row gather below.
    if (op_case == 10) {
        // Reshape to [1,T,K,1] for the broadcast-multiply with experts [1,T,K,n_embd]. Use an
        // explicit [1,-1,K,1] reshape (K is static, read via PartialShape to avoid .to_shape()
        // throwing when T is dynamic) instead of Squeeze+Unsqueeze, which the CPU plugin
        // implements as a Reshape internally and mis-infers the static pattern when T=1.
        const int64_t K = context.get_output_shape()[2].get_length();
        auto idx = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
        auto ge = std::make_shared<ov::op::v6::GatherElements>(data, idx, -1);  // [1,1,T,K]
        auto col = std::make_shared<ov::op::v1::Reshape>(
            ge,
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, K, 1}),
            false);  // [1,T,K,1]
        return rename_outputs_with_suffix({col}, context.get_name());
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        indices = process_view_input(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    auto indices_raw = indices;
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    // Axis to re-insert the size-1 axis dropped above: 0 for the batch_dims/inp_out_ids cases
    // below, 2 for the plain embedding lookup (see its branch for why).
    int64_t restore_axis = 0;
    // inp_out_ids picks specific whole rows (usually just the last token); MoE/test-backend-ops
    // gathers pick different indices per row (batch_dims=1). Data's own shape can't tell these
    // apart anymore once its leading axis becomes backend-dependent (see below), so match by name.
    const bool is_output_row_select = context.get_input_names()[1] == "inp_out_ids";
    if (data.get_partial_shape().rank() == 4) {
        if (is_output_row_select) {
            // Flatten to [rows, hidden] first: data's leading axes carry the same batch-vs-tokens
            // ambiguity fixed for the embedding lookup below, so a fixed Squeeze(axes=[0,1]) (as
            // this used to do) breaks depending on backend. Reshape doesn't reorder memory, so this
            // is correct either way. Keep indices rank-2 ([rows, 1]) so Gather's output stays rank 3,
            // matching what the Unsqueeze below expects.
            const int64_t hidden = data.get_partial_shape()[3].get_length();
            auto data_flat = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, hidden}),
                false);
            auto indices_flat = std::make_shared<ov::op::v1::Reshape>(
                indices_raw,
                ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, 1}),
                false);
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data_flat, indices_flat, axis);
        } else {
            // Per-row (batch_dims=1) selection: axis 0 isn't GGUF's batch/token axis here, so no
            // backend-dependent handling needed.
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else {
        // Embedding-table lookup: Gather's output shape mirrors indices' own shape ([1, tokens]
        // pre-PA, [tokens, 1] post-PA -- SDPAToPagedAttention rewrites input_ids to rank-1), so its
        // leading axis is self-correcting: 1 (batch) pre-PA, tokens post-PA -- what
        // PagedAttentionExtension expects Q/K/V's leading axis to hold. Restore the rank-4 form at
        // axis 2 (not 0) so that axis isn't overwritten with a literal 1.
        restore_axis = 2;
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    // The two Squeezes above dropped the leading axes; restore ggml's rank-4 form.
    res = std::make_shared<ov::op::v0::Unsqueeze>(res,
                                                  ov::op::v0::Constant::create(ov::element::i64, {1}, {restore_axis}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
