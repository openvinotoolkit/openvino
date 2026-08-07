// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_split_kv_fusion.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include <vector>

namespace ov::intel_gpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace {
// The Softmax/Concat of the split-attention pattern must operate on the last axis. Returns false on
// dynamic rank (a non-static graph never reaches the split-KV path anyway).
bool is_last_axis(const std::shared_ptr<const ov::Node>& node, int64_t axis) {
    const auto rank = node->get_output_partial_shape(0).rank();
    if (rank.is_dynamic()) {
        return false;
    }
    const auto r = rank.get_length();
    return (axis < 0 ? axis + r : axis) == r - 1;
}
}  // namespace

SDPASplitKVFusion::SDPASplitKVFusion() {
    // Anchored on the output Add of the two V-matmuls. The structural checks below are verified
    // explicitly in the callback.
    auto output_add = wrap_type<v1::Add>({wrap_type<v0::MatMul>(), wrap_type<v0::MatMul>()});

    ov::matcher_pass_callback callback = [](Matcher& m) {
        auto add_node = ov::as_type_ptr<v1::Add>(m.get_match_root());
        if (!add_node) {
            return false;
        }

        // output = Add(MatMul(pc, V_cache), MatMul(pn, V_new))
        auto matmul_cache = ov::as_type_ptr<v0::MatMul>(add_node->input_value(0).get_node_shared_ptr());
        auto matmul_new = ov::as_type_ptr<v0::MatMul>(add_node->input_value(1).get_node_shared_ptr());
        if (!matmul_cache || !matmul_new) {
            return false;
        }

        // Both V matmuls' probability input must come from the same VariadicSplit (outputs 0 and 1).
        // Note: paired Slice nodes are folded to VariadicSplit by GroupedSliceToVSplitOptimization in
        // CommonOptimizations before this pass runs.
        auto cache_src = matmul_cache->input_value(0);
        auto new_src = matmul_new->input_value(0);
        auto vsplit_node = ov::as_type_ptr<v1::VariadicSplit>(cache_src.get_node_shared_ptr());
        if (!vsplit_node || vsplit_node != ov::as_type_ptr<v1::VariadicSplit>(new_src.get_node_shared_ptr())) {
            return false;
        }
        if (cache_src.get_index() != 0 || new_src.get_index() != 1) {
            return false;
        }

        auto softmax_node = ov::as_type_ptr<v8::Softmax>(vsplit_node->input_value(0).get_node_shared_ptr());
        if (!softmax_node || !is_last_axis(softmax_node, softmax_node->get_axis())) {
            return false;
        }

        auto mask_add_node = ov::as_type_ptr<v1::Add>(softmax_node->input_value(0).get_node_shared_ptr());
        if (!mask_add_node) {
            return false;
        }

        auto concat_node = ov::as_type_ptr<v0::Concat>(mask_add_node->input_value(0).get_node_shared_ptr());
        if (!concat_node || concat_node->get_input_size() != 2 || !is_last_axis(concat_node, concat_node->get_axis())) {
            return false;
        }

        auto qk_cache_node = ov::as_type_ptr<v0::MatMul>(concat_node->input_value(0).get_node_shared_ptr());
        auto qk_new_node = ov::as_type_ptr<v0::MatMul>(concat_node->input_value(1).get_node_shared_ptr());
        if (!qk_cache_node || !qk_new_node) {
            return false;
        }

        // Both QK matmuls must share Q and agree on transpose flags; same for the V matmuls.
        if (qk_cache_node->input_value(0) != qk_new_node->input_value(0)) {
            return false;
        }
        if (qk_cache_node->get_transpose_a() || qk_new_node->get_transpose_a()) {
            return false;
        }
        if (qk_cache_node->get_transpose_b() != qk_new_node->get_transpose_b()) {
            return false;
        }
        if (matmul_cache->get_transpose_a() || matmul_new->get_transpose_a()) {
            return false;
        }
        if (matmul_cache->get_transpose_b() != matmul_new->get_transpose_b()) {
            return false;
        }

        auto Q = qk_cache_node->input_value(0);
        auto K_cache = qk_cache_node->input_value(1);
        auto K_new = qk_new_node->input_value(1);
        auto V_cache = matmul_cache->input_value(1);
        auto V_new = matmul_new->input_value(1);
        auto mask_value = mask_add_node->input_value(1);

        // Require fully static 4D shapes. The split-KV op is lowered only by the partitioned opt
        // decode path (SDPAOpt's single-token generators with SPLIT_KV, which need static shapes to
        // size their SLM scores[] and partition bounds); there is no dynamic-shape split-KV kernel,
        // so a dynamic match would have no implementation. Dynamic graphs fall back to the regular
        // (non-split) SDPA path.
        for (const auto& in : {Q, K_cache, K_new, V_cache, V_new}) {
            const auto& ps = in.get_partial_shape();
            if (ps.is_dynamic() || ps.rank().get_length() != 4) {
                return false;
            }
        }

        // ---- GQA un-fold (Q only) --------------------------------------------------------------
        // The Gemma split-attention export folds the GQA group (q_heads / kv_heads) into Q's query
        // axis, so Q reaches the QK MatMul as [B, kv_heads, group*S_q, D] through a Reshape (in
        // prefill preceded by a Transpose). That producer chain is NOT matched here (the pattern is
        // anchored on the output Add and the QK MatMul's Q operand is read structurally), so we walk
        // up from the already-matched Q to read its shape hints. Feeding the folded Q straight to the
        // split-KV op makes heads == kv_heads with an inflated q_len, and the kernel dispatches its
        // work-group grid over (batch*kv_heads, group*S_q) instead of (batch*q_heads, S_q) -- decode
        // gets misclassified as multi-token prefill. Recover the canonical Q [B, q_heads, S_q, D] (a
        // pure reshape of the folded Q: the group is the outer part of the folded query axis, folded
        // row f*S_q + s -> head k*group+f, seq s). K/V stay SPLIT with kv_heads -- the split-KV kernel
        // broadcasts them up to q_heads itself (DO_BROADCAST_KEY_VALUE), so unlike a naive concat-based
        // fusion we do not Concat/Broadcast K/V here. Any assumption that does not hold falls through
        // to the folded inputs below, which stay numerically correct.
        ov::Output<ov::Node> q_sdpa = Q;
        ov::Output<ov::Node> mask_sdpa = mask_value;
        bool refolded = false;  // true once Q is un-folded to canonical [B, q_heads, S_q, D]
        {
            const auto q_ps = Q.get_partial_shape();
            const auto kc_head_ps = K_cache.get_partial_shape();
            if (q_ps.is_static() && kc_head_ps[1].is_static()) {
                const int64_t B = q_ps[0].get_length();
                const int64_t kvh = kc_head_ps[1].get_length();
                const int64_t F = q_ps[2].get_length();  // group * S_q
                const int64_t D = q_ps[3].get_length();

                // Recover (q_heads, S_q) by inspecting Q's producer (outside the matched pattern):
                //  (a) a Reshape (prefill: preceded by a Transpose) whose projection is
                //      [B, S_q, q_heads, D] -- read q_heads / S_q straight off it; or
                //  (b) no fold Reshape at all, which the export emits only when the fold is an
                //      identity, i.e. kvh == 1 and S_q == 1, so q_heads == F.
                int64_t qh = 0, s_q = 0;
                if (auto fold_reshape = ov::as_type_ptr<v1::Reshape>(Q.get_node_shared_ptr())) {
                    ov::Output<ov::Node> proj = fold_reshape->input_value(0);
                    if (auto tr = ov::as_type_ptr<v1::Transpose>(proj.get_node_shared_ptr())) {
                        proj = tr->input_value(0);
                    }
                    const auto proj_ps = proj.get_partial_shape();
                    if (proj_ps.rank().is_static() && proj_ps.size() == 4 && proj_ps[1].is_static() &&
                        proj_ps[2].is_static()) {
                        s_q = proj_ps[1].get_length();
                        qh = proj_ps[2].get_length();
                    }
                } else if (q_ps[1].is_static() && q_ps[1].get_length() == 1 && kvh == 1) {
                    // Folded Q is the projection itself ([B, 1, q_heads, D] with S_q == 1).
                    qh = F;
                    s_q = 1;
                }

                if (qh > 0 && kvh > 0 && qh > kvh && qh % kvh == 0 && s_q > 0 && (qh / kvh) * s_q == F) {
                    // Canonical Q: pure reshape [B, kvh, group*S_q, D] -> [B, q_heads, S_q, D].
                    auto q_tgt = v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{B, qh, s_q, D});
                    q_sdpa = std::make_shared<v1::Reshape>(Q, q_tgt, false);
                    refolded = true;

                    // The mask [B, 1, group*S_q, S_kv] is group-replicated along the query axis (it
                    // depends on key/query position, not head), so the first S_q rows are the
                    // canonical per-sequence mask, broadcastable across q_heads.
                    const auto m_ps = mask_value.get_partial_shape();
                    if (F != s_q && m_ps.rank().is_static() && m_ps.size() == 4 && m_ps[2].is_static() &&
                        m_ps[2].get_length() == F) {
                        auto start = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{0});
                        auto stop = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{s_q});
                        auto step = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{1});
                        auto axis = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{2});
                        mask_sdpa = std::make_shared<v8::Slice>(mask_value, start, stop, step, axis);
                    }
                }
            }
        }

        // Only the default contiguous K/V layout is supported (K and V both stored [B,H,S,D], seq
        // axis = 2). The split-KV kernel reads K/V with the head dim contiguous; transposed K/V
        // ([B,H,D,S]) is intentionally NOT handled here -- such graphs fall back to the regular
        // (non-split) SDPA path. The transpose_b flags encode the physical layout (see common
        // matcher): qk transpose_b == true  -> K is [B,H,S,D] (what we want);
        //           attn transpose_b == false -> V is [B,H,S,D] (what we want).
        if (!qk_cache_node->get_transpose_b() || matmul_cache->get_transpose_b()) {
            return false;
        }
        const int64_t seq_axis = 2;  // K/V seq axis for the contiguous [B,H,S,D] layout

        // Decode only (q_len == 1). The split-KV path is the SPLIT_KV-gated branch of sdpa_opt's
        // single-token decode kernel (see sdpa_opt.cl, #ifdef SPLIT_KV). Prefill / multi-token chunks
        // (q_len > 1) fall through to the regular (non-split) SDPA path. The query length is read from
        // the fused op's Q (q_sdpa): the canonical S_q after the GQA un-fold, else the folded Q's
        // dim 2. The new-chunk seq lengths must be STATIC (the kernel sizes its SLM scores[] and
        // partition bounds on SOURCE_SEQ_LEN at compile time).
        const auto& q_sdpa_ps = q_sdpa.get_partial_shape();
        if (!q_sdpa_ps[seq_axis].is_static() || q_sdpa_ps[seq_axis].get_length() != 1) {
            return false;
        }
        const auto& k_new_ps = K_new.get_partial_shape();
        const auto& v_new_ps = V_new.get_partial_shape();
        if (!k_new_ps[seq_axis].is_static() || !v_new_ps[seq_axis].is_static()) {
            return false;
        }

        // Derive the valid cache length from the additive mask, as the trailing kv_len input so the
        // kernel can cap its cache loops (the cache is allocated for the full context but in decode
        // only the first `time_step+1` positions are real; the rest are padding the mask drives to
        // exp(-100)~=0). Compute valid_cache_len = (last attended cache index) + 1. Using the LAST-
        // attended index (not the zero count) keeps this correct for BOTH global (causal prefix
        // [0,t]) and local sliding-window (window [t-w, t]) masks: the bound is a superset of the
        // attended set, and any masked interior position still gets the additive mask -> exp~=0.
        //
        //   keep   = Equal(mask_cache, 0)        ; bool [B,1,q,S_cache]
        //   idx1   = Convert(keep, i32) * ramp   ; i32  [B,1,q,S_cache]   ramp = [1..S_cache]
        //   kv_len = ReduceMax(idx1, all axes)   ; i32  scalar = last_attended + 1
        // The mask spans [cache | new]; slice off the cache columns first. S_cache is static (the
        // fusion requires fully static shapes), so the ramp constant is well-defined.
        const int64_t s_cache = K_cache.get_partial_shape()[2].get_length();  // [B,H,S,D]
        auto sl_start = v0::Constant::create(ov::element::i64, {1}, {0});
        auto sl_stop = v0::Constant::create(ov::element::i64, {1}, {s_cache});
        auto sl_step = v0::Constant::create(ov::element::i64, {1}, {1});
        auto sl_axis = v0::Constant::create(ov::element::i64, {1}, {3});
        auto mask_cache = std::make_shared<v8::Slice>(mask_value, sl_start, sl_stop, sl_step, sl_axis);

        auto zero = v0::Constant::create(mask_value.get_element_type(), {}, {0.0f});
        auto keep = std::make_shared<v1::Equal>(mask_cache, zero);
        auto keep_i = std::make_shared<v0::Convert>(keep, ov::element::i32);
        std::vector<int32_t> ramp_vals(static_cast<size_t>(s_cache));
        for (int64_t i = 0; i < s_cache; ++i)
            ramp_vals[static_cast<size_t>(i)] = static_cast<int32_t>(i + 1);  // 1-based
        auto ramp = v0::Constant::create(ov::element::i32, ov::Shape{static_cast<size_t>(s_cache)}, ramp_vals);
        auto idx1 = std::make_shared<v1::Multiply>(keep_i, ramp);
        auto axes = v0::Constant::create(ov::element::i64, {4}, {0, 1, 2, 3});
        auto kv_len = std::make_shared<v1::ReduceMax>(idx1, axes, /*keep_dims=*/false);

        // Build the split-KV op inputs [Q, K_cache, V_cache, mask, K_new, V_new, kv_len]. No scale
        // INPUT: the matched pattern has no scale node and any scaling is baked into Q (e.g. Gemma's
        // query_pre_attn_scalar), so the lowering sets scale_val = 1.0 and the kernel applies it via
        // STATIC_SCALE_VALUE. K/V stay split -- the kernel broadcasts them across heads. Q and mask
        // carry the GQA un-fold (q_sdpa / mask_sdpa), falling back to the folded Q / full mask when
        // the un-fold did not apply. Layout is always default contiguous [B,H,S,D].
        ov::OutputVector inputs{q_sdpa, K_cache, V_cache, mask_sdpa, K_new, V_new, kv_len};
        const auto order = op::SDPA::default_order(/*rank=*/4);
        auto sdpa = std::make_shared<op::SDPA>(inputs,
                                               /*is_causal=*/false,
                                               order,
                                               order,
                                               order,
                                               order,
                                               ov::element::dynamic,
                                               /*split_kv=*/true);

        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        // The split-KV op emits the canonical attention output [B, q_heads, S_q, D]. When Q was
        // un-folded above, the matched Add (the graph's consumer/output) still expects the folded
        // GQA layout [B, kvh, group*S_q, D]. Re-fold the canonical output back with the inverse pure
        // reshape to add_node's own static output shape so downstream shapes are unchanged; without
        // it the replacement node carries the canonical shape and breaks the model's output binding.
        // When no un-fold happened the SDPA output already matches add_node, so replace directly.
        ov::Output<ov::Node> result = sdpa;
        if (refolded) {
            const auto& out_shape = add_node->get_output_shape(0);  // static (checked above)
            auto out_tgt = v0::Constant::create(ov::element::i64, {out_shape.size()},
                                                std::vector<int64_t>(out_shape.begin(), out_shape.end()));
            result = std::make_shared<v1::Reshape>(sdpa, out_tgt, false);
        }
        result.get_node_shared_ptr()->set_friendly_name(add_node->get_friendly_name());
        ov::replace_node(add_node, result.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<Matcher>(output_add, "SDPASplitKVFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
