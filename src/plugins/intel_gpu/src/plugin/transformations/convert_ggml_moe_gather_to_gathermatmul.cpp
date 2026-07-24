// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_ggml_moe_gather_to_gathermatmul.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "transformations/pattern_blocks/compressed_weights_block.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
// [dims...] of a ShapeOf, as an i64 1-D tensor (helper mirrors the frontend's get_dimensions).
std::shared_ptr<ov::Node> gather_dims(const ov::Output<ov::Node>& shape_of, const std::vector<int64_t>& dims) {
    auto idx = ov::op::v0::Constant::create(ov::element::i64, {dims.size()}, dims);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    return std::make_shared<ov::op::v8::Gather>(shape_of, idx, axis);
}
}  // namespace

ConvertGgmlMoeGatherToGatherMatmul::ConvertGgmlMoeGatherToGatherMatmul() {
    using namespace ov::pass::pattern;

    // The ggml-openvino frontend lowers the gemma-style top-k expert matmul (MUL_MAT_ID)
    // using PUBLIC ops only, as a rank-2 carrier:
    //
    //   weights = CompressedWeightsBlock -> [n_expert, m*k]  (low-bit Constant -> Convert
    //             -> [Subtract(zp)] -> Multiply(scale) -> Reshape([n_expert, m*k]) -> [Convert])
    //   sel     = Gather(weights, ids[n_tokens, n_used], axis=0)   -> [n_tokens, n_used, m*k]
    //   [sel    = Convert(sel, f32)]
    //   selr    = Reshape(sel, [n_tokens, n_used, m, -1])          -> [n_tokens, n_used, m, k]
    //   acts    = ... -> [n_tokens, n_used, 1, k]                  (broadcast over experts)
    //   result  = MatMul(acts, selr, transpose_b=true)            -> [n_tokens, n_used, 1, m]
    //   result  = Reshape(result, [1, n_tokens, n_used, m])
    //
    // The frontend keeps the expert Gather on the rank-2 [n_expert, m*k] block on purpose:
    // that folds to a plain GatherCompressed on ANY OpenVINO build (weights stay compressed,
    // no OOM), so the frontend is safe even without this pass.
    //
    // When this pass IS present (GPU, systolic + oneDNN MoE pipeline), we can do better:
    // rewrite the carrier to the internal ov::op::internal::GatherMatmul so the downstream
    // ConvertGatherMatmulToGatherMatmulCompressed folds it into a single
    // GatherMatmulCompressed (gather + dequantize-selected-experts + matmul in one op).
    // GatherMatmul needs a rank-3 [n_expert, N, K] weights block, so we rebuild the block's
    // low-bit / scale / zero-point Constants at the grouped rank-4 [n_expert, N, K/group,
    // group] shape (a bit-identical reinterpret of the same data) and reshape to rank-3.
    //
    // Requiring a genuine CompressedWeightsBlock as the Gather's source both guarantees the
    // downstream fold can fire and structurally excludes any non-compressed expert matmul.
    const std::vector<ov::element::Type> supported_weights_types{ov::element::u4, ov::element::i4, ov::element::u8,
                                                                 ov::element::i8};
    m_weights_block =
        std::make_shared<ov::pass::pattern::op::CompressedWeightsBlock>(supported_weights_types, std::set<size_t>{2});
    auto weights_block = m_weights_block;
    auto ids_m = any_input();
    auto gather_m = wrap_type<ov::op::v8::Gather>({weights_block, ids_m, any_input()}, consumers_count(1));
    auto conv_m = wrap_type<ov::op::v0::Convert>({gather_m}, consumers_count(1));
    auto wsel_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{gather_m, conv_m});
    // The frontend reshapes the gathered flat weights [n_tokens, n_used, m*k] back to
    // [n_tokens, n_used, m, k] (a plain Reshape, not a Split op). Its target-shape input is
    // built dynamically (Concat of ShapeOf gathers), NOT a Constant, so it must be
    // matched as any_input().
    auto weights_reshape_m = wrap_type<ov::op::v1::Reshape>({wsel_m, any_input()}, consumers_count(1));
    auto acts_m = any_input();
    auto matmul_m = wrap_type<ov::op::v0::MatMul>({acts_m, weights_reshape_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_root());
        if (!matmul || !matmul->get_transpose_b() || matmul->get_transpose_a()) {
            return false;
        }
        if (pm.count(gather_m) == 0 || pm.count(ids_m) == 0 || pm.count(acts_m) == 0) {
            return false;
        }

        // --- Extract the compressed-weight Constants from the matched block --------------
        auto block = std::static_pointer_cast<ov::pass::pattern::op::CompressedWeightsBlock>(m_weights_block);
        auto w_anchor = block->get_anchor("weights", pm);
        auto s_anchor = block->get_anchor("mul_const", pm);
        if (!w_anchor.has_value() || !s_anchor.has_value()) {
            return false;
        }
        auto w_const = ov::as_type_ptr<ov::op::v0::Constant>(w_anchor->get_node_shared_ptr());
        auto s_const = ov::as_type_ptr<ov::op::v0::Constant>(s_anchor->get_node_shared_ptr());
        if (!w_const || !s_const) {
            return false;
        }
        // Zero point is optional (symmetric i4/i8 experts have none).
        std::shared_ptr<ov::op::v0::Constant> zp_const;
        if (auto sub_a = block->get_anchor("sub_const", pm); sub_a.has_value()) {
            zp_const = ov::as_type_ptr<ov::op::v0::Constant>(sub_a->get_node_shared_ptr());
            if (!zp_const) {
                return false;  // a Subtract is present but its operand isn't a Constant we can regroup
            }
        }

        // The frontend's rank-2 weight Constant is grouped as [n_expert, (m*k)/group, group].
        const auto w_shape = w_const->get_shape();
        if (w_shape.size() != 3) {
            return false;
        }
        const int64_t n_expert = static_cast<int64_t>(w_shape[0]);
        const int64_t mkg = static_cast<int64_t>(w_shape[1]);  // (m*k)/group
        const int64_t group = static_cast<int64_t>(w_shape[2]);

        // N (= m, per-expert output rows) is the static row dim of the MatMul output
        // [n_tokens, n_used, 1, m].
        const auto mm_out = matmul->get_output_partial_shape(0);
        if (mm_out.rank().is_dynamic() || mm_out.rank().get_length() != 4 || mm_out[3].is_dynamic()) {
            return false;
        }
        const int64_t N = mm_out[3].get_length();
        if (N <= 0 || mkg % N != 0) {
            return false;
        }
        const int64_t k_blk = mkg / N;   // per-row block count
        const int64_t K = k_blk * group;  // per-expert input width

        // The scale (and zp, if present) are grouped as [n_expert, (m*k)/group, 1]; their
        // element counts must match the regrouped [n_expert, N, k_blk, 1] layout.
        const int64_t grouped_scale_elems = n_expert * N * k_blk;
        if (static_cast<int64_t>(ov::shape_size(s_const->get_shape())) != grouped_scale_elems) {
            return false;
        }
        if (zp_const && static_cast<int64_t>(ov::shape_size(zp_const->get_shape())) != grouped_scale_elems) {
            return false;
        }

        // ids -> [n_tokens, n_used].
        ov::Output<ov::Node> idx = pm.at(ids_m);
        if (idx.get_partial_shape().rank().is_dynamic() || idx.get_partial_shape().rank().get_length() != 2) {
            return false;
        }
        if (idx.get_element_type() != ov::element::i32 && idx.get_element_type() != ov::element::i64) {
            idx = std::make_shared<ov::op::v0::Convert>(idx, ov::element::i32);
        }

        // --- Rebuild the compressed weights as a rank-3 [n_expert, N, K] block -----------
        // Reinterpret the SAME low-bit data at the grouped rank-4 shape [n_expert, N, k_blk,
        // group] (bit-identical: (m*k)/group * group == N * k_blk * group), then run the
        // dequant chain and reshape to rank-3, matching CompressedWeightsBlock so the
        // downstream compressed fold fires.
        const ov::Shape w4_shape{static_cast<size_t>(n_expert), static_cast<size_t>(N), static_cast<size_t>(k_blk),
                                 static_cast<size_t>(group)};
        const ov::Shape grouped_scale_shape{static_cast<size_t>(n_expert), static_cast<size_t>(N),
                                             static_cast<size_t>(k_blk), 1};

        auto w4 = std::make_shared<ov::op::v0::Constant>(w_const->get_element_type(), w4_shape, w_const->get_data_ptr());
        auto s4 = std::make_shared<ov::op::v0::Constant>(s_const->get_element_type(), grouped_scale_shape,
                                                         s_const->get_data_ptr());
        auto w_f16 = std::make_shared<ov::op::v0::Convert>(w4, ov::element::f16);
        std::shared_ptr<ov::Node> dequant;
        if (zp_const) {
            auto zp4 = std::make_shared<ov::op::v0::Constant>(zp_const->get_element_type(), grouped_scale_shape,
                                                              zp_const->get_data_ptr());
            std::shared_ptr<ov::Node> zp_f16 = zp4;
            if (zp4->get_element_type() != ov::element::f16) {
                zp_f16 = std::make_shared<ov::op::v0::Convert>(zp4, ov::element::f16);
            }
            auto w_zp = std::make_shared<ov::op::v1::Subtract>(w_f16, zp_f16, ov::op::AutoBroadcastType::NUMPY);
            dequant = std::make_shared<ov::op::v1::Multiply>(w_zp, s4, ov::op::AutoBroadcastType::NUMPY);
        } else {
            dequant = std::make_shared<ov::op::v1::Multiply>(w_f16, s4, ov::op::AutoBroadcastType::NUMPY);
        }
        auto w3_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, {n_expert, N, K});
        ov::Output<ov::Node> w3d = std::make_shared<ov::op::v1::Reshape>(dequant, w3_shape, false);
        // Keep the dequant chain from being const-folded before the compressed fold runs.
        ov::pass::disable_constant_folding(w_f16);
        ov::pass::disable_constant_folding(dequant);
        ov::pass::disable_constant_folding(w3d.get_node_shared_ptr());

        // --- Recover the pre-broadcast activations as GatherMatmul input A ---------------
        // The frontend feeds MatMul input0 as acts broadcast to [n_tokens, n_used, 1, k].
        // GatherMatmul wants A = [batch(1 or n_used), n_tokens, k]:
        //   shared (up/gate, acts dim1 == 1)      -> [1, n_tokens, k]
        //   per-expert (down, acts dim1 == n_used) -> transpose to [n_used, n_tokens, k]
        ov::Output<ov::Node> acts = pm.at(acts_m);
        const auto acts_ps = acts.get_partial_shape();
        if (acts_ps.rank().is_dynamic() || acts_ps.rank().get_length() != 4) {
            return false;
        }
        auto acts_shape = std::make_shared<ov::op::v3::ShapeOf>(acts, ov::element::i64);
        const bool per_expert = acts_ps[1].is_static() && acts_ps[1].get_length() > 1;
        ov::Output<ov::Node> A;
        if (per_expert) {
            auto sq_dims = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{gather_dims(acts_shape, {0}), gather_dims(acts_shape, {1}),
                                 gather_dims(acts_shape, {3})},
                0);
            auto acts3 = std::make_shared<ov::op::v1::Reshape>(acts, sq_dims, false);
            auto perm = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2});
            A = std::make_shared<ov::op::v1::Transpose>(acts3, perm);
        } else {
            auto a_dims = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), gather_dims(acts_shape, {0}),
                                 gather_dims(acts_shape, {3})},
                0);
            A = std::make_shared<ov::op::v1::Reshape>(acts, a_dims, false);
        }
        if (A.get_element_type() != ov::element::f32) {
            A = std::make_shared<ov::op::v0::Convert>(A, ov::element::f32);
        }

        auto gm = std::make_shared<ov::op::internal::GatherMatmul>(A, w3d, idx);  // [n_used, n_tokens, N]

        // [n_used, n_tokens, N] -> [1, n_tokens, n_used, N] to match the original MatMul output.
        auto perm2 = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 0, 2});
        ov::Output<ov::Node> out = std::make_shared<ov::op::v1::Transpose>(gm, perm2);  // [n_tokens, n_used, N]
        auto gm_shape = std::make_shared<ov::op::v3::ShapeOf>(out, ov::element::i64);
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto out_dims = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, gm_shape}, 0);
        out = std::make_shared<ov::op::v1::Reshape>(out, out_dims, false);
        if (out.get_element_type() != matmul->get_output_element_type(0)) {
            out = std::make_shared<ov::op::v0::Convert>(out, matmul->get_output_element_type(0));
        }

        out.get_node_shared_ptr()->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), out.get_node_shared_ptr());
        ov::replace_node(matmul, out.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_m, "ConvertGgmlMoeGatherToGatherMatmul");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
