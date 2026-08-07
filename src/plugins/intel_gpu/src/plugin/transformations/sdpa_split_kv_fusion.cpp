// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_split_kv_fusion.hpp"

#include <vector>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;

using ov::pass::pattern::any_input;
using ov::pass::pattern::consumers_count;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::rank_equals;
using ov::pass::pattern::wrap_type;

namespace {
// The K/V Concat of the (already core-fused) split-attention pattern must operate on the sequence
// axis of the canonical [B,H,S,D] layout, i.e. rank-2 (not rank-1, which is the head dim D). Returns
// false on dynamic rank (a non-static graph never reaches the split-KV path anyway).
bool is_seq_axis(const std::shared_ptr<const ov::Node>& node, int64_t axis) {
    const auto rank = node->get_output_partial_shape(0).rank();
    if (rank.is_dynamic()) {
        return false;
    }
    const auto r = rank.get_length();
    return (axis < 0 ? axis + r : axis) == r - 2;
}
}  // namespace

SDPASplitKVFusion::SDPASplitKVFusion() {
    // Anchored on the fused v13::ScaledDotProductAttention that ov::pass::SDPASplitAttentionFusionMatcher
    // (registered immediately before this pass in the GPU pipeline -- see transformations_pipeline.cpp)
    // already builds from the raw split-attention sub-graph:
    //
    //   sdpa = v13::SDPA(Q, Concat(K_cache, K_new, seq_axis), Concat(V_cache, V_new, seq_axis), mask, scale=1.0)
    //
    // Concatenating the whole KV cache into a single tensor every decode step would copy it (the
    // cache is a host-owned Parameter that cannot be concatenated in place), so this pass un-does
    // the K/V concat for GPU and keeps K_cache/K_new and V_cache/V_new as separate op::SDPA inputs
    // (split_kv = true) instead. The structural checks below are verified explicitly in the callback.
    auto q_input = any_input(rank_equals(4));
    auto k_cache_input = any_input(rank_equals(4));
    auto k_new_input = any_input(rank_equals(4));
    auto v_cache_input = any_input(rank_equals(4));
    auto v_new_input = any_input(rank_equals(4));
    auto mask_input = any_input();
    auto scale_input = any_input();

    auto k_concat = wrap_type<v0::Concat>({k_cache_input, k_new_input}, consumers_count(1));
    auto v_concat = wrap_type<v0::Concat>({v_cache_input, v_new_input}, consumers_count(1));
    auto sdpa_m = wrap_type<v13::ScaledDotProductAttention>({q_input, k_concat, v_concat, mask_input, scale_input});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto sdpa_node = ov::as_type_ptr<v13::ScaledDotProductAttention>(m.get_match_root());
        if (!sdpa_node || sdpa_node->get_causal()) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto k_concat_node = ov::as_type_ptr<v0::Concat>(pattern_map.at(k_concat).get_node_shared_ptr());
        auto v_concat_node = ov::as_type_ptr<v0::Concat>(pattern_map.at(v_concat).get_node_shared_ptr());
        if (!k_concat_node || !v_concat_node) {
            return false;
        }

        // Only the default contiguous [B,H,S,D] layout is supported: the core fusion concats
        // directly on the sequence axis (no intervening Transpose) only for this layout; a
        // transposed K/V ([B,H,D,S]) shows up here as Transpose(Concat(...)) instead of a bare
        // Concat feeding the SDPA, which this pattern does not match, so such graphs fall through
        // to the regular (non-split) SDPA path untouched.
        if (!is_seq_axis(k_concat_node, k_concat_node->get_axis()) ||
            !is_seq_axis(v_concat_node, v_concat_node->get_axis())) {
            return false;
        }

        // The split-KV kernel hardcodes STATIC_SCALE_VALUE = 1.0 (there is no scale input); the core
        // fusion always emits exactly this constant (any scaling is assumed pre-baked into Q), so
        // require it here too -- a different scale value has no split-KV kernel to lower to.
        auto scale_const = ov::as_type_ptr<v0::Constant>(pattern_map.at(scale_input).get_node_shared_ptr());
        if (!scale_const) {
            return false;
        }
        const auto scale_vals = scale_const->cast_vector<double>();
        if (scale_vals.size() != 1 || scale_vals[0] != 1.0) {
            return false;
        }

        auto Q = pattern_map.at(q_input);
        auto K_cache = pattern_map.at(k_cache_input);
        auto K_new = pattern_map.at(k_new_input);
        auto V_cache = pattern_map.at(v_cache_input);
        auto V_new = pattern_map.at(v_new_input);
        auto mask_value = pattern_map.at(mask_input);

        // The kv_len derivation further below assumes the canonical rank-4 mask [B, 1, q, S_kv]
        // (it slices off the cache columns at a hardcoded axis 3). mask_input has no rank
        // constraint in the pattern (a lower-rank mask broadcastable to the QK scores is legal per
        // the SDPA spec), so bail to the regular (non-split) SDPA path rather than build a Slice
        // with an out-of-range axis.
        const auto mask_ps = mask_value.get_partial_shape();
        if (!mask_ps.rank().is_static() || mask_ps.size() != 4) {
            return false;
        }

        // Require fully static 4D shapes (rank 4 is already guaranteed by the pattern above). The
        // split-KV op is lowered only by the partitioned opt decode path (SDPAOpt's single-token
        // generators with SPLIT_KV, which need static shapes to size their SLM scores[] and
        // partition bounds); there is no dynamic-shape split-KV kernel, so a dynamic match would
        // have no implementation. Dynamic graphs fall back to the regular (non-split) SDPA path.
        for (const auto& in : {Q, K_cache, K_new, V_cache, V_new}) {
            if (in.get_partial_shape().is_dynamic()) {
                return false;
            }
        }

        // ---- GQA un-fold (Q only) --------------------------------------------------------------
        // The Gemma split-attention export folds the GQA group (q_heads / kv_heads) into Q's query
        // axis, so Q reaches the fused SDPA as [B, kv_heads, group*S_q, D] through a Reshape (in
        // prefill preceded by a Transpose). That producer chain is NOT matched here (the pattern is
        // anchored on the fused SDPA and Q is read structurally as its 1st input), so we walk
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
                    // canonical per-sequence mask, broadcastable across q_heads. mask_ps is already
                    // known rank-4 (checked above).
                    if (F != s_q && mask_ps[2].is_static() && mask_ps[2].get_length() == F) {
                        auto start = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{0});
                        auto stop = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{s_q});
                        auto step = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{1});
                        auto axis = v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{2});
                        mask_sdpa = std::make_shared<v8::Slice>(mask_value, start, stop, step, axis);
                    }
                }
            }
        }

        // K and V are both stored [B,H,S,D] (seq axis = 2): the bare-Concat check above already
        // guarantees the contiguous layout, since a transposed K/V would route through a Transpose
        // node instead of feeding the SDPA concat directly (see the pattern comment above).
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
        // un-folded above, the matched SDPA (the graph's consumer/output) still expects the folded
        // GQA layout [B, kvh, group*S_q, D]. Re-fold the canonical output back with the inverse pure
        // reshape to sdpa_node's own static output shape so downstream shapes are unchanged; without
        // it the replacement node carries the canonical shape and breaks the model's output binding.
        // When no un-fold happened the SDPA output already matches sdpa_node, so replace directly.
        ov::Output<ov::Node> result = sdpa;
        if (refolded) {
            const auto& out_shape = sdpa_node->get_output_shape(0);  // static (checked above)
            auto out_tgt = v0::Constant::create(ov::element::i64, {out_shape.size()},
                                                std::vector<int64_t>(out_shape.begin(), out_shape.end()));
            result = std::make_shared<v1::Reshape>(sdpa, out_tgt, false);
        }
        result.get_node_shared_ptr()->set_friendly_name(sdpa_node->get_friendly_name());
        ov::replace_node(sdpa_node, result.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<Matcher>(sdpa_m, "SDPASplitKVFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
