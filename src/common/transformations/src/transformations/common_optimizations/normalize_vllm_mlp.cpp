// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/normalize_vllm_mlp.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// vLLM lowers a fused gate_up MLP block in two possible forms:
//   Form A: split via two Slice ops on the same source.
//   Form B: split via a VariadicSplit but with i64 split_lengths or with a
//           positive-axis (axis=1 for [B*S, H]) instead of axis=-1.
// CPU LLMMLPFusion requires a VariadicSplit with i32 lengths shape [2] and
// axis = -1 (literal). This pass canonicalizes both forms so that fusion
// can collapse Swish(gate) * up + down_proj into a single LLMMLP primitive.
NormalizeVLLMMLP::NormalizeVLLMMLP() {
    MATCHER_SCOPE(NormalizeVLLMMLP);
    using namespace pattern;

    auto callback = [=](Matcher& m) -> bool {
        auto root = m.get_match_root();
        auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(root);
        if (!mul) return false;

        std::shared_ptr<ov::op::v4::Swish> swish;
        std::shared_ptr<ov::op::v7::Gelu> gelu;
        std::shared_ptr<ov::Node> activation;  // either Swish or Gelu
        std::shared_ptr<ov::op::v8::Slice> up_slice;
        std::shared_ptr<ov::op::v1::VariadicSplit> up_vsplit;
        ov::Output<ov::Node> up_out;
        for (size_t i = 0; i < 2; ++i) {
            auto sw = std::dynamic_pointer_cast<ov::op::v4::Swish>(mul->get_input_node_shared_ptr(i));
            auto ge = std::dynamic_pointer_cast<ov::op::v7::Gelu>(mul->get_input_node_shared_ptr(i));
            std::shared_ptr<ov::Node> act = sw ? std::static_pointer_cast<ov::Node>(sw)
                                                : std::static_pointer_cast<ov::Node>(ge);
            if (!act) continue;
            auto other = mul->input_value(1 - i);
            auto sl = std::dynamic_pointer_cast<ov::op::v8::Slice>(other.get_node_shared_ptr());
            auto vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(other.get_node_shared_ptr());
            if (sl) { swish = sw; gelu = ge; activation = act; up_slice = sl; break; }
            if (vs) { swish = sw; gelu = ge; activation = act; up_vsplit = vs; up_out = other; break; }
        }
        if (!activation || (!up_slice && !up_vsplit)) return false;
        auto _make_act = [&](const ov::Output<ov::Node>& in) -> std::shared_ptr<ov::Node> {
            if (gelu) {
                return std::make_shared<ov::op::v7::Gelu>(in, gelu->get_approximation_mode());
            }
            return std::make_shared<ov::op::v4::Swish>(in);
        };

        // Branch B: already a VariadicSplit. Canonicalize lengths to i32 [2]
        // and axis to -1 if needed. Also elide the narrow-residual Convert
        // pair (f32->bf16 before VariadicSplit, bf16->f32 after Multiply)
        // that vLLM emits for bf16 model dtype. Removing the pair skips a
        // bf16 round-trip that LLMMLPFusion would otherwise have to tolerate
        // via optional-Convert relaxations in intel_cpu/mlp_fusion.cpp.
        if (up_vsplit) {
            auto sw_in = activation->input_value(0);
            auto gate_vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(sw_in.get_node_shared_ptr());
            if (!gate_vs || gate_vs.get() != up_vsplit.get()) return false;

            auto sl_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(up_vsplit->get_input_node_shared_ptr(2));
            if (!sl_const) return false;
            auto et = sl_const->get_element_type();
            auto shp = sl_const->get_shape();
            bool lens_ok = (et == ov::element::i32 && shp.size() == 1 && shp[0] == 2);

            auto axis_const_chk = std::dynamic_pointer_cast<ov::op::v0::Constant>(up_vsplit->get_input_node_shared_ptr(1));
            bool axis_is_neg_one = false;
            if (axis_const_chk) {
                auto av = axis_const_chk->cast_vector<int64_t>();
                if (!av.empty() && av[0] == -1) axis_is_neg_one = true;
            }

            // Detect narrow-Convert wedged between the gate_up MatMul (f32)
            // and this VariadicSplit. In vLLM bf16 graphs it takes the form
            // `MatMul(f32) -> Convert(f32->bf16) -> VariadicSplit(bf16)`.
            // Bypass it if present so LLMMLPFusion's f32 pattern matches.
            ov::Output<ov::Node> new_vs_data = up_vsplit->input_value(0);
            auto pre_cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(
                new_vs_data.get_node_shared_ptr());
            bool bypassed_pre_cvt = false;
            if (pre_cvt) {
                auto src = pre_cvt->input_value(0);
                auto src_t = src.get_element_type();
                auto dst_t = pre_cvt->get_destination_type();
                if ((dst_t == ov::element::bf16 || dst_t == ov::element::f16) &&
                    src_t == ov::element::f32) {
                    new_vs_data = src;
                    bypassed_pre_cvt = true;
                }
            }

            // If we bypassed the pre-Convert, we also need the gate_up MatMul
            // weight Constant to match intel_cpu LLMMLPFusion's f16 predicate.
            // vLLM stores weights in the model's native dtype (bf16 for
            // Llama-3.2-family); recast statically to fp16 so the pattern
            // fires. bf16 has narrower mantissa (7 vs f16's 10 bits) but
            // wider exponent (8 vs 5). MLP weight magnitudes are <<1 so
            // fp16 range (max 65504) is not a concern. Precision widens
            // slightly (7 mantissa -> 10 mantissa).
            std::shared_ptr<ov::op::v0::Constant> new_gate_up_weight_const;
            std::shared_ptr<ov::op::v0::Convert> old_weight_cvt;
            if (bypassed_pre_cvt) {
                // Walk pre-cvt (which we bypassed above) -> MatMul -> weight
                // Convert -> weight Constant.
                auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                    pre_cvt->input_value(0).get_node_shared_ptr());
                if (mm) {
                    auto wcvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(
                        mm->input_value(1).get_node_shared_ptr());
                    if (wcvt) {
                        auto wcst = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                            wcvt->input_value(0).get_node_shared_ptr());
                        if (wcst && wcst->get_element_type() == ov::element::bf16 &&
                            wcvt->get_destination_type() == ov::element::f32) {
                            // Static bf16 -> fp16 recast (lossless in range for
                            // MLP weights). Use Constant::create<float16> from
                            // upcast-to-f32 vector.
                            auto vals = wcst->cast_vector<float>();
                            new_gate_up_weight_const = ov::op::v0::Constant::create(
                                ov::element::f16, wcst->get_shape(), vals);
                            old_weight_cvt = wcvt;
                        }
                    }
                }
            }

            if (lens_ok && axis_is_neg_one && !bypassed_pre_cvt) return false;

            auto vals = sl_const->cast_vector<int64_t>();
            if (vals.size() != 2) return false;

            auto axis_const_new = ov::op::v0::Constant::create(
                ov::element::i32, ov::Shape{}, {static_cast<int32_t>(-1)});
            auto split_lengths_new = ov::op::v0::Constant::create(
                ov::element::i32, ov::Shape{2},
                {static_cast<int32_t>(vals[0]), static_cast<int32_t>(vals[1])});
            // If we rewrote the weight, redirect the MatMul's weight Convert
            // input to the new f16 Constant. The Convert becomes f16 -> f32,
            // matching intel_cpu's `wrap_type<Convert>(gate_up_proj_weight)`.
            if (new_gate_up_weight_const && old_weight_cvt) {
                auto new_wcvt = std::make_shared<ov::op::v0::Convert>(
                    new_gate_up_weight_const, ov::element::f32);
                for (auto& c : old_weight_cvt->output(0).get_target_inputs()) {
                    c.replace_source_output(new_wcvt->output(0));
                }
            }

            auto new_vsplit = std::make_shared<ov::op::v1::VariadicSplit>(
                new_vs_data, axis_const_new, split_lengths_new);

            size_t gate_idx = sw_in.get_index();
            size_t up_idx = up_out.get_index();
            if (gate_idx == up_idx) up_idx = 1 - gate_idx;

            auto new_swish = _make_act(new_vsplit->output(gate_idx));
            auto new_mul = std::make_shared<ov::op::v1::Multiply>(
                new_swish->output(0), new_vsplit->output(up_idx));
            new_mul->set_friendly_name(mul->get_friendly_name());
            ov::copy_runtime_info({up_vsplit, activation, mul}, {new_vsplit, new_swish, new_mul});
            ov::replace_node(mul, new_mul);

            // Elide any post-Multiply Convert(bf16/f16 -> f32) that vLLM's
            // narrow-residual graph inserts before down_proj. Route those
            // consumers back to new_mul directly. Multi-Convert consumers
            // are handled by walking new_mul's target inputs; other
            // non-Convert consumers (e.g. residual add expecting narrow
            // dtype) are left untouched.
            std::shared_ptr<ov::op::v0::MatMul> down_proj_mm;
            for (auto& consumer : new_mul->output(0).get_target_inputs()) {
                auto cvt = ov::as_type<ov::op::v0::Convert>(consumer.get_node());
                if (!cvt) continue;
                auto cvt_src = new_mul->output(0).get_element_type();
                auto cvt_dst = cvt->get_destination_type();
                // Only elide the narrow-then-wide sequence: mul (f32) skips
                // the redundant Convert(f32->f32) that appears after we've
                // bypassed the pre-Convert. If the source is truly narrow
                // (bf16/f16) then the round-trip is real and we leave it.
                if (cvt_src == cvt_dst) {
                    for (auto& downstream : cvt->output(0).get_target_inputs()) {
                        downstream.replace_source_output(new_mul->output(0));
                    }
                }
            }

            // Rank-2 wrap: if the gate_up MatMul source is rank-2 (vLLM's
            // flattened [B*S, H]), the intel_cpu MLPFusion pattern which
            // requires rank-3 [B, S, H] activation won't match. Insert
            // Unsqueeze(axis=0) at the gate_up MatMul's activation input
            // and Squeeze(axis=0) after the down_proj MatMul so the
            // interior chain (MatMul + VariadicSplit + Swish + Multiply +
            // MatMul) flows as rank-3 [1, B*S, H]. Non-MLP consumers of
            // the shared source and of the down_proj output continue to
            // see the rank-2 tensor.
            auto gate_up_mm_walk = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                new_vs_data.get_node_shared_ptr());
            if (gate_up_mm_walk) {
                auto shared_src = gate_up_mm_walk->input_value(0);
                auto src_ps_rk2 = shared_src.get_partial_shape();
                if (src_ps_rk2.rank().is_static() && src_ps_rk2.rank().get_length() == 2) {
                    // Locate down_proj MatMul reached from new_mul (possibly
                    // through an intermediate Convert).
                    auto walk_to_matmul = [](const ov::Output<ov::Node>& out)
                            -> std::shared_ptr<ov::op::v0::MatMul> {
                        for (auto& consumer : out.get_target_inputs()) {
                            auto node = consumer.get_node()->shared_from_this();
                            if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node)) {
                                return mm;
                            }
                            if (std::dynamic_pointer_cast<ov::op::v0::Convert>(node)) {
                                for (auto& c2 : node->output(0).get_target_inputs()) {
                                    auto n2 = c2.get_node()->shared_from_this();
                                    if (auto mm2 = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n2)) {
                                        return mm2;
                                    }
                                }
                            }
                        }
                        return nullptr;
                    };
                    down_proj_mm = walk_to_matmul(new_mul->output(0));
                    if (down_proj_mm) {
                        auto out_ps_rk2 = down_proj_mm->output(0).get_partial_shape();
                        if (out_ps_rk2.rank().is_static() &&
                            out_ps_rk2.rank().get_length() == 2) {
                            auto zero_axis = ov::op::v0::Constant::create(
                                ov::element::i32, ov::Shape{1}, {0});
                            auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(
                                shared_src, zero_axis);
                            gate_up_mm_walk->input(0).replace_source_output(unsqueezed->output(0));

                            auto zero_axis_sq = ov::op::v0::Constant::create(
                                ov::element::i32, ov::Shape{1}, {0});
                            auto squeezed = std::make_shared<ov::op::v0::Squeeze>(
                                down_proj_mm->output(0), zero_axis_sq);
                            for (auto& c : down_proj_mm->output(0).get_target_inputs()) {
                                if (c.get_node() == squeezed.get()) continue;
                                c.replace_source_output(squeezed->output(0));
                            }
                            ov::copy_runtime_info(ov::NodeVector{gate_up_mm_walk}, unsqueezed);
                            ov::copy_runtime_info(ov::NodeVector{down_proj_mm}, squeezed);
                        }
                    }
                }
            }
            return true;
        }

        // Branch A: two Slice ops on the same source.
        auto gate_slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(activation->get_input_node_shared_ptr(0));
        if (!gate_slice) return false;
        auto gate_src = gate_slice->input_value(0);
        auto up_src = up_slice->input_value(0);
        if (gate_src != up_src) return false;

        auto src_ps = gate_src.get_partial_shape();
        if (!src_ps.rank().is_static()) return false;
        auto r = src_ps.rank().get_length();
        if (r < 1) return false;
        if (!src_ps[r - 1].is_static()) return false;
        int64_t full = src_ps[r - 1].get_length();
        if (full <= 0 || full % 2 != 0) return false;
        int64_t half = full / 2;

        auto check_slice = [&](const std::shared_ptr<ov::op::v8::Slice>& s,
                               int64_t expect_start, int64_t expect_stop) -> bool {
            if (s->get_input_size() < 5) return false;
            auto starts = std::dynamic_pointer_cast<ov::op::v0::Constant>(s->get_input_node_shared_ptr(1));
            auto stops  = std::dynamic_pointer_cast<ov::op::v0::Constant>(s->get_input_node_shared_ptr(2));
            auto steps  = std::dynamic_pointer_cast<ov::op::v0::Constant>(s->get_input_node_shared_ptr(3));
            auto axes   = std::dynamic_pointer_cast<ov::op::v0::Constant>(s->get_input_node_shared_ptr(4));
            if (!starts || !stops || !steps || !axes) return false;
            auto sv = starts->cast_vector<int64_t>();
            auto ev = stops->cast_vector<int64_t>();
            auto stv = steps->cast_vector<int64_t>();
            auto av = axes->cast_vector<int64_t>();
            if (sv.size() != 1 || ev.size() != 1 || stv.size() != 1 || av.size() != 1) return false;
            if (stv[0] != 1) return false;
            int64_t ax = av[0]; if (ax < 0) ax += r;
            if (ax != r - 1) return false;
            int64_t stop_val = ev[0];
            if (stop_val > full) stop_val = full;
            if (sv[0] != expect_start || stop_val != expect_stop) return false;
            return true;
        };

        bool order_normal = check_slice(gate_slice, 0, half) && check_slice(up_slice, half, full);
        bool order_swapped = !order_normal &&
                             check_slice(gate_slice, half, full) && check_slice(up_slice, 0, half);
        if (!order_normal && !order_swapped) return false;

        auto axis_const = ov::op::v0::Constant::create(
            ov::element::i32, ov::Shape{}, {static_cast<int32_t>(-1)});
        auto split_lengths = ov::op::v0::Constant::create(
            ov::element::i32, ov::Shape{2},
            {static_cast<int32_t>(half), static_cast<int32_t>(half)});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(gate_src, axis_const, split_lengths);

        size_t gate_idx = order_normal ? 0 : 1;
        size_t up_idx   = order_normal ? 1 : 0;

        auto new_swish = _make_act(vsplit->output(gate_idx));
        auto new_mul = std::make_shared<ov::op::v1::Multiply>(new_swish->output(0), vsplit->output(up_idx));
        new_mul->set_friendly_name(mul->get_friendly_name());
        ov::copy_runtime_info({gate_slice, up_slice, activation, mul},
                              {vsplit, new_swish, new_mul});
        ov::replace_node(mul, new_mul);
        return true;
    };

    auto mul_pattern = wrap_type<ov::op::v1::Multiply>();
    auto m = std::make_shared<Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
