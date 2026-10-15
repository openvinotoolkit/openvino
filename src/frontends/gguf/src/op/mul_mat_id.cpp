// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> slice_axis(const ov::Output<ov::Node>& input, int64_t axis, int64_t begin, int64_t end) {
    return std::make_shared<ov::op::v8::Slice>(input,
                                               const_i64({begin}),
                                               const_i64({end}),
                                               const_i64({1}),
                                               const_i64({axis}));
}

// Lay the 3D activations out as [n_token, n_used, k]: one row per (token, selected expert) pair,
// which is exactly the batch layout of the gathered expert weights, so the MatMul below needs no
// implicit batch broadcasting.
//
// ggml hands us the activations of a MUL_MAT_ID in one of three layouts, all three of which occur in
// a single MoE model:
//   [n_token, 1,       k]  the usual gate/up input -- a singleton expert axis to expand;
//   [n_token, n_used,  k]  the ffn_down input, whose expert axis is already expanded;
//   [1,       n_token, k]  the plain 2D form, with no expert axis at all, so the token count sits one
//                          axis over. It appears in the last layer, where the activations are first
//                          gathered down to the ubatch's output rows.
// Only the second may not be reshaped -- it already holds n_token * n_used * k elements, so folding it
// to one row per token would drop data. For the other two the Reshape below is what puts the token
// count in axis 0, and it is a no-op for the first, so the two need no telling apart. The second is
// recognized by its expert axis matching the ids' n_used, which holds on a static (NPU) graph too,
// where no axis is dynamic to give the layout away.
//
// Driving that Reshape from the *ids'* token count is also what makes an empty ubatch work: the
// output-row count is 0 for every non-final chunk of a chunked prefill (only the final chunk produces
// logits), and 0 tokens must stay 0 rows. Broadcasting an empty axis up to n_used is rejected by
// OpenVINO -- and meaningless, as there is nothing to replicate -- whereas here the empty count simply
// flows into the token axis and yields an empty result.
ov::Output<ov::Node> activations_per_expert(const ov::Output<ov::Node>& activations,
                                            const std::shared_ptr<ov::op::v3::ShapeOf>& activations_shape,
                                            const ov::Output<ov::Node>& ids,
                                            const std::shared_ptr<ov::op::v3::ShapeOf>& ids_shape) {
    const auto& acts_ps = activations.get_partial_shape();  // [n_token, 1 | n_used, k], or [1, n_token, k]
    const auto& ids_ps = ids.get_partial_shape();           // [n_token, n_used]
    // The expert-axis match alone is ambiguous: the plain 2D layout [1, n_token, k] also satisfies it
    // when n_token == n_used (the warmup graph). The token axis (1 vs n_token) disambiguates.
    const bool already_per_expert = acts_ps.rank().is_static() && acts_ps.rank().get_length() == 3 &&
                                    ids_ps.rank().is_static() && ids_ps.rank().get_length() == 2 &&
                                    acts_ps[1].is_static() && ids_ps[1].is_static() && ids_ps[1].get_length() > 1 &&
                                    acts_ps[1] == ids_ps[1] &&
                                    (acts_ps[0].is_dynamic() || ids_ps[0].is_dynamic() || acts_ps[0] == ids_ps[0]);

    ov::Output<ov::Node> rows = activations;
    if (!already_per_expert) {
        rows = std::make_shared<ov::op::v1::Reshape>(
            activations,
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{get_dimensions(ids_shape, {0}),
                                                                  const_i64({1}),
                                                                  get_dimensions(activations_shape, {2})},
                                                 0),
            false);
    }
    auto target = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), get_dimensions(activations_shape, {2})},
        0);
    return std::make_shared<ov::op::v3::Broadcast>(rows, target, ov::op::BroadcastType::BIDIRECTIONAL);
}

// Packed-MXFP4 MoE path: expert weights arrive as raw u8 blocks of shape
// [1, n_expert, m, k_blocks, 17] (1 e8m0 scale byte + 16 nibble-packed f4e2m1 quants per 32-block).
// The dequant (nibble unpack + LUT + scale) is done on-graph, per selected expert.
ov::Output<ov::Node> translate_mul_mat_id_mxfp4_packed(const NodeContext& context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    auto packed_shape = expert_weights.get_partial_shape().to_shape();
    FRONT_END_OP_CONVERSION_CHECK(packed_shape.size() == 5 && packed_shape[4] == 17,
                                  "Expected packed MXFP4 expert weights with shape [1, n_expert, m, k_blocks, 17]");

    const int64_t n_expert = static_cast<int64_t>(packed_shape[1]);
    const int64_t rows = static_cast<int64_t>(packed_shape[2]);
    const int64_t k_blocks = static_cast<int64_t>(packed_shape[3]);
    const int64_t qk = 32;
    const int64_t cols = k_blocks * qk;

    auto packed_shape_4d = const_i64({n_expert, rows, k_blocks, 17});
    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, packed_shape_4d, false);

    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);
    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    static const std::vector<float> f4e2m1_lut =
        {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    // Input-independent e8m0 exponent table; build once (was rebuilt per invocation).
    static const std::vector<float> e8m0_lut = [] {
        std::vector<float> lut(256);
        for (size_t i = 0; i < lut.size(); ++i) {
            uint32_t bits = static_cast<uint32_t>(i) << 23;
            memcpy(&lut[i], &bits, sizeof(float));
        }
        lut[0] = std::numeric_limits<float>::min() / 2.0f;
        lut[255] = std::numeric_limits<float>::quiet_NaN();
        return lut;
    }();

    auto f4_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{f4e2m1_lut.size()}, f4e2m1_lut);
    auto scale_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{e8m0_lut.size()}, e8m0_lut);

    auto selected_packed_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
    auto scale_byte = slice_axis(selected_packed_weights, 4, 0, 1);
    auto qs = slice_axis(selected_packed_weights, 4, 1, 17);
    auto low =
        std::make_shared<ov::op::v13::BitwiseAnd>(qs,
                                                  ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {0x0F}),
                                                  ov::op::AutoBroadcastType::NUMPY);
    auto high_shift = std::make_shared<ov::op::v15::BitwiseRightShift>(
        qs,
        ov::op::v0::Constant::create(ov::element::u8, ov::Shape{}, {4}),
        ov::op::AutoBroadcastType::NUMPY);
    auto nibbles = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{low, high_shift}, 4);
    auto nibble_indices = std::make_shared<ov::op::v0::Convert>(nibbles, ov::element::i32);
    auto weights_f32 = std::make_shared<ov::op::v8::Gather>(f4_lut, nibble_indices, gather_axis);

    auto scale_indices = std::make_shared<ov::op::v0::Convert>(scale_byte, ov::element::i32);
    auto scales_f32 = std::make_shared<ov::op::v8::Gather>(scale_lut, scale_indices, gather_axis);
    ov::Output<ov::Node> selected_weights =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::AutoBroadcastType::NUMPY);

    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto selected_weights_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{get_dimensions(ids_shape, {0, 1}), const_i64({rows, cols})},
        0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, selected_weights_target_dims, false);

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto acts_broadcasted = activations_per_expert(activations, activations_shape, ids, ids_shape);

    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, const_i64({2}));
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {rows});
    auto result_target_dims =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim},
                                             0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    const auto output_type = context.get_attribute<ov::element::Type>("output_type");
    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    return result;
}

// Lower to the internal ov::op::internal::GatherMatmul, which the CPU/GPU plugins execute as one
// optimized batched expert-matmul (and, when the expert weights are a compressed
// Constant->Convert->[Subtract]->Multiply block, fold into GatherMatmulCompressed so the weights
// stay compressed -- no host f32 expansion, which is what keeps MoE compile memory bounded). The
// CPU GatherMatmul node requires CONSTANT-backed weights, so this path is used only when the
// expert weights come from a Constant (the real .gguf builder / cgraph weight leaf). GatherMatmul:
//   A       [n_activated, T, cols]   (n_activated == 1 broadcasts the same input to every
//                                      selected expert; == K gives a per-slot input)
//   B       [n_expert, rows, cols]   (transpose_b=true -> A . Bᵀ)
//   indices [T, K] i32               (the selected expert per (token, slot))
//   out     [K, T, rows]
// which we reshape back to the builder's [1, T, K, rows] convention.
//
// Input layouts (OpenVINO reversed order):
//   expert_weights (as) : [n_expert, rows, cols]  (or reversed rank-4 [1, n_expert, rows, cols])
//   activations (b)     : [.., T, cols] (gate/up, shared input) or [.., T, K, cols] (down)
//   ids                 : [1, 1, T, K]
ov::Output<ov::Node> translate_mul_mat_id_gathermatmul(const NodeContext& context,
                                                       ov::Output<ov::Node> expert_weights,
                                                       ov::Output<ov::Node> activations,
                                                       ov::Output<ov::Node> ids) {
    // Normalize the expert weights to the rank-3 [n_expert, rows, cols] GatherMatmul expects. The
    // native builder surfaces them rank-3 already; the cgraph path surfaces them as a reversed
    // rank-4 [1, n_expert, rows, cols], so drop the leading unit batch dim.
    ov::Output<ov::Node> as = expert_weights;
    if (as.get_partial_shape().rank().is_static() && as.get_partial_shape().rank().get_length() == 4) {
        auto as_shape = std::make_shared<ov::op::v3::ShapeOf>(as, ov::element::i64);
        as = std::make_shared<ov::op::v1::Reshape>(as, get_dimensions(as_shape, {1, 2, 3}), false);
    }
    auto b = activations;

    // Canonicalize ids to 2D [T, K] (the builder carries leading 1-dims).
    const auto ids_rank = static_cast<int>(ids.get_partial_shape().size());
    ov::Output<ov::Node> ids_2d = std::make_shared<ov::op::v1::Reshape>(
        ids,
        std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                             get_dimensions(ids, {ids_rank - 1})},
            0),
        false);  // [T, K]
    const int64_t K = ids.get_partial_shape()[ids_rank - 1].get_length();

    // Build the GatherMatmul activation A = [n_activated, T, cols].
    //   gate/up: b is [.., T, cols] (one shared input fanned out to all experts) -> A = [1, T, cols]
    //   down   : b is [.., T, K, cols] (already per-slot)                         -> A = [K, T, cols]
    const auto& bps = b.get_partial_shape();
    const int64_t cols = bps[bps.size() - 1].get_length();
    const bool has_k =
        K > 1 && bps.size() >= 2 && bps[bps.size() - 2].is_static() && bps[bps.size() - 2].get_length() == K;
    ov::Output<ov::Node> a;
    if (has_k) {
        auto b_tkc = std::make_shared<ov::op::v1::Reshape>(
            b,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, K, cols}),
            false);
        a = std::make_shared<ov::op::v1::Transpose>(
            b_tkc,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2}));
    } else {
        a = std::make_shared<ov::op::v1::Reshape>(
            b,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, -1, cols}),
            false);
    }

    if (ids_2d.get_element_type() != ov::element::i32) {
        ids_2d = std::make_shared<ov::op::v0::Convert>(ids_2d, ov::element::i32);
    }

    // B stays in its (possibly compressed) precision so ConvertGatherMatmulToGatherMatmulCompressed
    // can pick up the decompression subgraph and keep the weights compressed.
    auto gmm = std::make_shared<ov::op::internal::GatherMatmul>(a, as, ids_2d);  // [K, T, rows]

    // [K, T, rows] -> [1, T, K, rows] (builder convention).
    const int64_t rows = as.get_partial_shape()[as.get_partial_shape().size() - 2].get_length();
    auto kt2tk = std::make_shared<ov::op::v1::Transpose>(
        gmm,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2}));  // [T, K, rows]
    return std::make_shared<ov::op::v1::Reshape>(
        kt2tk,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, -1, K, rows}),
        false);
}

// Portable fallback: per-token expert matmul via Gather (select each token's expert rows) + a
// batched MatMul. Handles non-constant expert weights (e.g. the single-op unit tests) that the CPU
// GatherMatmul node does not support. Expects reversed rank-4 inputs
// (weights [1, n_expert, m, k], activations [1, T, 1_or_K, k], ids [1, 1, T, K]).
ov::Output<ov::Node> translate_mul_mat_id_generic(const NodeContext& context,
                                                  ov::Output<ov::Node> expert_weights,
                                                  ov::Output<ov::Node> activations,
                                                  ov::Output<ov::Node> ids) {
    auto expert_weights_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(expert_weights, ov::element::i64);
    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights,
                                                           get_dimensions(expert_weights_shape_4d, {1, 2, 3}),
                                                           false);
    activations =
        std::make_shared<ov::op::v1::Reshape>(activations, get_dimensions(activations_shape_4d, {1, 2, 3}), false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, get_dimensions(ids_shape_4d, {2, 3}), false);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);

    const auto output_type = context.get_attribute<ov::element::Type>("output_type");
    if (selected_weights.get_element_type() != ov::element::f32) {
        selected_weights = std::make_shared<ov::op::v0::Convert>(selected_weights, ov::element::f32);
    }
    if (activations.get_element_type() != ov::element::f32) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, ov::element::f32);
    }

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto acts_broadcasted = activations_per_expert(activations, activations_shape, ids, ids_shape);

    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, unsqueeze_axes);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto output_shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(output_shape.rank().is_static() && output_shape.rank().get_length() == 4,
                                  "Unexpected MUL_MAT_ID output rank");
    FRONT_END_OP_CONVERSION_CHECK(output_shape[3].is_static(), "Expected static row dimension for MUL_MAT_ID output");
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3].get_length()});

    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);
    auto result_target_dims =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_dim, get_dimensions(ids_shape, {0, 1}), row_dim},
                                             0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);
    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }
    return result;
}

}  // namespace

// GGML_OP_MUL_MAT_ID: per-token MoE expert matmul. ids select which expert row of the weight
// tensor each token uses; activations are gathered/broadcast accordingly and matmul'd. Dispatch:
//   - packed MXFP4 experts -> on-graph dequant + per-expert matmul;
//   - constant-backed experts (the real .gguf models) -> internal GatherMatmul (compressed,
//     memory-bounded, one fused batched matmul);
//   - non-constant experts (single-op tests / dynamic) -> portable Gather + MatMul fallback.
OutputVector translate_mul_mat_id(const NodeContext& context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = context.get_input(0);
    auto activations = context.get_input(1);
    auto ids = context.get_input(2);

    if (expert_weights.get_element_type() == ov::element::u8 && expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() == 5) {
        return rename_outputs_with_suffix(
            {translate_mul_mat_id_mxfp4_packed(context, expert_weights, activations, ids)},
            context.get_name());
    }

    // The CPU GatherMatmul node requires constant-backed weights. Real .gguf models feed a
    // (possibly compressed) Constant weight leaf -> use the fused GatherMatmul path (compressed,
    // memory-bounded). A non-constant weights input (single-op tests, dynamic producers) can't use
    // GatherMatmul on CPU, so fall back to the portable Gather + MatMul lowering.
    ov::Output<ov::Node> result;
    if (ov::op::util::is_on_path<ov::op::v0::Constant>(expert_weights)) {
        result = translate_mul_mat_id_gathermatmul(context, expert_weights, activations, ids);
    } else {
        result = translate_mul_mat_id_generic(context, expert_weights, activations, ids);
    }
    return rename_outputs_with_suffix({std::move(result)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
