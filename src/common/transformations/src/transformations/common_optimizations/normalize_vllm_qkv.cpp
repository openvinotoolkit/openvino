// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/normalize_vllm_qkv.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// vLLM emits the QKV projection as
//   MatMul([B*S, H], W^T) -> [Convert(bf16->f32)] -> VariadicSplit -> q,k,v
// with rank-2 input, axis Constant of i64 (literal -1), and split-lengths
// Constant of i64 [3]. CPU QKVProjFusion was originally written for rank-3
// activations, axis=2 literal, and i32 split-lengths. This pass canonicalizes
// those mismatches so the existing pattern can match without rank or dtype
// concessions.
//
// Three rewrites, all metadata-only:
//   1. Wrap the input in Reshape -> [1, B*S, H] so the activation is rank-3.
//      The MatMul output gets re-flattened with Reshape -> [B*S, len_i] for
//      downstream consumers.
//   2. Replace the VariadicSplit axis Constant with literal i32 axis=2.
//   3. Cast the split_lengths Constant to i32.
//
// This pass leaves the asymmetric Q/K/V split lengths (GQA: Hq != Hk) intact;
// the CPU QKVProjFusion callback still has to accept asymmetric sizes (one
// narrow plugin-side relaxation that no graph rewrite can replace).
//
// Anchor matches the VariadicSplit. Walking back through the optional
// post-MatMul Convert and the MatMul itself is done in the callback so we
// can reuse the existing producer if it already feeds a Convert.
NormalizeVLLMQKV::NormalizeVLLMQKV() {
    MATCHER_SCOPE(NormalizeVLLMQKV);
    using namespace pattern;

    auto vsplit_p = wrap_type<ov::op::v1::VariadicSplit>();

    auto callback = [=](Matcher& m) -> bool {
        auto vs = ov::as_type_ptr<ov::op::v1::VariadicSplit>(m.get_match_root());
        if (!vs) return false;

        auto axis_c = ov::as_type_ptr<ov::op::v0::Constant>(vs->get_input_node_shared_ptr(1));
        auto lens_c = ov::as_type_ptr<ov::op::v0::Constant>(vs->get_input_node_shared_ptr(2));
        if (!axis_c || !lens_c) return false;

        auto av = axis_c->cast_vector<int64_t>();
        if (av.size() != 1) return false;
        auto lv = lens_c->cast_vector<int64_t>();
        if (lv.size() != 3) return false;
        for (auto x : lv) if (x <= 0) return false;

        // Walk: VariadicSplit -> [Convert] -> MatMul -> activation
        auto vs_src = vs->input_value(0);
        std::shared_ptr<ov::op::v0::Convert> post_cvt;
        std::shared_ptr<ov::op::v0::MatMul> mm;
        auto producer = vs_src.get_node_shared_ptr();
        if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(producer)) {
            post_cvt = cvt;
            producer = cvt->input_value(0).get_node_shared_ptr();
        }
        mm = ov::as_type_ptr<ov::op::v0::MatMul>(producer);
        if (!mm) return false;

        // Only kick in when the canonical CPU pattern can't match as-is:
        // either input is rank-2, or axis const isn't i32 == 2, or lens
        // const isn't i32.
        auto act = mm->input_value(0);
        auto act_ps = act.get_partial_shape();
        bool act_rank2 = act_ps.rank().is_static() && act_ps.rank().get_length() == 2;
        bool act_rank3 = act_ps.rank().is_static() && act_ps.rank().get_length() == 3;
        if (!act_rank2 && !act_rank3) return false;

        // Resolve the canonical positive last-dim index for the (post-wrap) rank.
        int64_t target_rank = act_rank2 ? 3 : act_ps.rank().get_length();
        int64_t axis_canonical = target_rank - 1;

        // Validate axis points to last dim.
        int64_t cur_rank = act_rank2 ? 2 : act_ps.rank().get_length();
        int64_t axis_val = av[0];
        if (axis_val < 0) axis_val += cur_rank;
        if (axis_val != cur_rank - 1) return false;

        bool axis_is_canonical_i32 =
            (axis_c->get_element_type() == ov::element::i32) &&
            axis_c->get_shape().empty() &&
            !axis_c->cast_vector<int64_t>().empty() &&
            axis_c->cast_vector<int64_t>()[0] == axis_canonical;
        bool lens_is_i32 =
            (lens_c->get_element_type() == ov::element::i32) &&
            lens_c->get_shape().size() == 1 && lens_c->get_shape()[0] == 3;

        if (axis_is_canonical_i32 && lens_is_i32 && act_rank3) return false;

        // 1. Wrap the activation in Unsqueeze axis=0 when rank-2 ([B*S, H] -> [1, B*S, H]).
        ov::Output<ov::Node> mm_input = act;
        std::shared_ptr<ov::Node> wrap_node;
        if (act_rank2) {
            auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
            wrap_node = std::make_shared<ov::op::v0::Unsqueeze>(act, axes);
            mm_input = wrap_node->output(0);
        }

        // Rebuild MatMul on the wrapped input.
        auto mm_new = std::make_shared<ov::op::v0::MatMul>(
            mm_input, mm->input_value(1),
            mm->get_transpose_a(), mm->get_transpose_b());

        // Rebuild post-MatMul Convert if present.
        ov::Output<ov::Node> vs_input = mm_new->output(0);
        if (post_cvt) {
            auto cvt_new = std::make_shared<ov::op::v0::Convert>(vs_input, post_cvt->get_destination_type());
            vs_input = cvt_new->output(0);
        }

        // 2 + 3. Replace axis with i32 canonical, cast lens to i32.
        auto axis_new = ov::op::v0::Constant::create(
            ov::element::i32, ov::Shape{}, {static_cast<int32_t>(axis_canonical)});
        auto lens_new = ov::op::v0::Constant::create(
            ov::element::i32, ov::Shape{3},
            {static_cast<int32_t>(lv[0]), static_cast<int32_t>(lv[1]), static_cast<int32_t>(lv[2])});
        auto vs_new = std::make_shared<ov::op::v1::VariadicSplit>(vs_input, axis_new, lens_new);

        // Each VariadicSplit output is (post-wrap) rank-3 [1, B*S, len_i];
        // consumers expect rank-2 [B*S, len_i] when the original input was
        // rank-2. Reflatten with Reshape -> [-1, len_i].
        for (size_t i = 0; i < 3; ++i) {
            ov::Output<ov::Node> out_i = vs_new->output(i);
            if (act_rank2) {
                auto flat_const = ov::op::v0::Constant::create(
                    ov::element::i32, ov::Shape{2},
                    {static_cast<int32_t>(-1), static_cast<int32_t>(lv[i])});
                auto r2 = std::make_shared<ov::op::v1::Reshape>(out_i, flat_const, false);
                out_i = r2->output(0);
            }
            vs->output(i).replace(out_i);
        }
        ov::NodeVector from_nodes{mm, vs};
        if (post_cvt) from_nodes.push_back(post_cvt);
        ov::copy_runtime_info(from_nodes, {mm_new, vs_new});
        return true;
    };

    auto m = std::make_shared<Matcher>(vsplit_p, "NormalizeVLLMQKV");
    register_matcher(m, callback);
}

}  // namespace ov::pass
