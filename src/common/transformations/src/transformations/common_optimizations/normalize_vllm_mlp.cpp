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
#include "openvino/op/swish.hpp"
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
        // and axis to -1 if needed.
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
            if (lens_ok && axis_is_neg_one) return false;

            auto vals = sl_const->cast_vector<int64_t>();
            if (vals.size() != 2) return false;

            auto axis_const_new = ov::op::v0::Constant::create(
                ov::element::i32, ov::Shape{}, {static_cast<int32_t>(-1)});
            auto split_lengths_new = ov::op::v0::Constant::create(
                ov::element::i32, ov::Shape{2},
                {static_cast<int32_t>(vals[0]), static_cast<int32_t>(vals[1])});
            auto new_vsplit = std::make_shared<ov::op::v1::VariadicSplit>(
                up_vsplit->input_value(0), axis_const_new, split_lengths_new);

            size_t gate_idx = sw_in.get_index();
            size_t up_idx = up_out.get_index();
            if (gate_idx == up_idx) up_idx = 1 - gate_idx;

            auto new_swish = _make_act(new_vsplit->output(gate_idx));
            auto new_mul = std::make_shared<ov::op::v1::Multiply>(
                new_swish->output(0), new_vsplit->output(up_idx));
            new_mul->set_friendly_name(mul->get_friendly_name());
            ov::copy_runtime_info({up_vsplit, activation, mul}, {new_vsplit, new_swish, new_mul});
            ov::replace_node(mul, new_mul);
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
