// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/normalize_vllm_rope.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::pass {

// vLLM lowers is_neox_style RoPE (post the common ConvertSubtract pass) as:
//   x1, x2 = split(x, axis=-1, num_splits=2)   // x1=split[0], x2=split[1]
//   o1 = Add(x1*cos, Multiply(x2*sin, -1))     // was Sub(x1*cos, x2*sin)
//   o2 = Add(x2*cos, x1*sin)
//   out = concat([o1, o2], axis=-1)
//
// The CPU/GPU plugin's RoPEFusionGPTNEOX matcher expects:
//   x_rot = concat([-x2, x1], axis=-1)
//   out = x * cos + x_rot * sin
//
// Mathematically equivalent. This pass rewrites the vLLM form so
// RoPEFusion can collapse it into a single RoPE primitive.
//
// Note: assumes ConvertSubtract has already run (which is true for all
// current CPU/GPU pipelines — it lives in ov::pass::CommonOptimizations).
NormalizeVLLMRoPE::NormalizeVLLMRoPE() {
    MATCHER_SCOPE(NormalizeVLLMRoPE);
    using namespace pattern;

    auto callback = [=](Matcher& m) -> bool {
        auto concat_node = std::dynamic_pointer_cast<ov::op::v0::Concat>(m.get_match_root());
        if (!concat_node) return false;
        if (concat_node->inputs().size() != 2) return false;

        // Concat inputs must be Add+Add: Add(x1*cos, (x2*sin)*-1) and Add(x2*cos, x1*sin).
        auto sub_branch = std::dynamic_pointer_cast<ov::op::v1::Add>(
            concat_node->get_input_node_shared_ptr(0));
        auto add_branch = std::dynamic_pointer_cast<ov::op::v1::Add>(
            concat_node->get_input_node_shared_ptr(1));
        if (!sub_branch || !add_branch) return false;

        // Find the negated Multiply (has a Constant -1 operand) inside sub_branch.
        auto is_neg_multiply = [](const std::shared_ptr<ov::Node>& n)
                                   -> std::shared_ptr<ov::op::v1::Multiply> {
            auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(n);
            if (!mul) return nullptr;
            for (size_t i = 0; i < 2; ++i) {
                auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(mul->get_input_node_shared_ptr(i));
                if (c) {
                    auto vals = c->cast_vector<float>();
                    if (!vals.empty() && vals[0] == -1.0f) return mul;
                }
            }
            return nullptr;
        };

        auto sub_in0 = sub_branch->get_input_node_shared_ptr(0);
        auto sub_in1 = sub_branch->get_input_node_shared_ptr(1);
        auto neg0 = is_neg_multiply(sub_in0);
        auto neg1 = is_neg_multiply(sub_in1);

        std::shared_ptr<ov::op::v1::Multiply> mul_a, mul_b, mul_c, mul_d;
        // mul_a = x1*cos (non-negated in sub_branch), mul_b = x2*sin (negated).
        if (neg0 && !neg1) {
            for (size_t i = 0; i < 2; ++i) {
                auto inner = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
                    neg0->get_input_node_shared_ptr(i));
                if (inner) { mul_b = inner; break; }
            }
            mul_a = std::dynamic_pointer_cast<ov::op::v1::Multiply>(sub_in1);
        } else if (neg1 && !neg0) {
            for (size_t i = 0; i < 2; ++i) {
                auto inner = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
                    neg1->get_input_node_shared_ptr(i));
                if (inner) { mul_b = inner; break; }
            }
            mul_a = std::dynamic_pointer_cast<ov::op::v1::Multiply>(sub_in0);
        } else {
            return false;
        }

        mul_c = std::dynamic_pointer_cast<ov::op::v1::Multiply>(add_branch->get_input_node_shared_ptr(0));
        mul_d = std::dynamic_pointer_cast<ov::op::v1::Multiply>(add_branch->get_input_node_shared_ptr(1));
        if (!mul_a || !mul_b || !mul_c || !mul_d) return false;

        // Trace each multiply's inputs back to a Split output (possibly through
        // view-changing ops Unsqueeze/Reshape/Squeeze). Return the split output
        // (with its .get_index() intact) and the "other side" (cos or sin).
        auto is_any_split = [](const std::shared_ptr<ov::Node>& n) {
            return std::dynamic_pointer_cast<ov::op::v1::Split>(n) != nullptr ||
                   std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(n) != nullptr;
        };
        auto find_split_source = [&is_any_split](ov::Output<ov::Node> val) -> ov::Output<ov::Node> {
            while (val.get_node()) {
                auto node = val.get_node_shared_ptr();
                if (is_any_split(node)) return val;
                if (std::dynamic_pointer_cast<ov::op::v0::Unsqueeze>(node) ||
                    std::dynamic_pointer_cast<ov::op::v1::Reshape>(node) ||
                    std::dynamic_pointer_cast<ov::op::v0::Squeeze>(node)) {
                    if (node->get_input_size() > 0) {
                        val = node->input_value(0);
                        continue;
                    }
                }
                break;
            }
            return ov::Output<ov::Node>();
        };
        auto classify = [&find_split_source](const std::shared_ptr<ov::op::v1::Multiply>& mul)
            -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
            auto in0 = mul->input_value(0);
            auto in1 = mul->input_value(1);
            auto s0 = find_split_source(in0);
            if (s0.get_node()) return {s0, in1};
            auto s1 = find_split_source(in1);
            if (s1.get_node()) return {s1, in0};
            return {ov::Output<ov::Node>(), ov::Output<ov::Node>()};
        };

        auto [a_split, a_other] = classify(mul_a);
        auto [b_split, b_other] = classify(mul_b);
        auto [c_split, c_other] = classify(mul_c);
        auto [d_split, d_other] = classify(mul_d);
        if (!a_split.get_node() || !b_split.get_node() ||
            !c_split.get_node() || !d_split.get_node()) return false;

        // All four multiplies must reference the same Split node.
        auto split_node = a_split.get_node_shared_ptr();
        if (b_split.get_node_shared_ptr() != split_node ||
            c_split.get_node_shared_ptr() != split_node ||
            d_split.get_node_shared_ptr() != split_node) return false;

        // Verify the split is 2-way and axis is the last dim.
        auto split_v1 = std::dynamic_pointer_cast<ov::op::v1::Split>(split_node);
        auto vsplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(split_node);
        if (!split_v1 && !vsplit) return false;
        if (split_v1 && split_v1->get_num_splits() != 2) return false;
        if (vsplit && vsplit->get_output_size() != 2) return false;

        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            split_node->get_input_node_shared_ptr(1));
        if (!axis_const) return false;
        auto axis_vec = axis_const->cast_vector<int64_t>();
        if (axis_vec.size() != 1) return false;
        auto rank = split_node->get_output_partial_shape(0).rank();
        if (!rank.is_static()) return false;
        auto r = rank.get_length();
        int64_t axis = axis_vec[0];
        if (axis < 0) axis += r;
        if (axis != r - 1) return false;

        // Verify (x1*cos, x2*sin, x2*cos, x1*sin) correspondence and shared cos/sin.
        if (a_split.get_index() != 0 || b_split.get_index() != 1) return false;
        if (c_split.get_index() != 1 || d_split.get_index() != 0) return false;
        if (a_other != c_other || b_other != d_other) return false;

        auto cos_val = a_other;
        auto sin_val = b_other;
        auto x_val = split_node->input_value(0);

        // If cos/sin are half-sized (match split output), build full-sized by
        // concatenating with themselves along the last dim.
        auto x_ps = x_val.get_partial_shape();
        auto cos_ps = cos_val.get_partial_shape();
        bool cos_needs_dup = false;
        if (x_ps.rank().is_static() && cos_ps.rank().is_static() &&
            x_ps.rank().get_length() == cos_ps.rank().get_length() &&
            x_ps.rank().get_length() > 0) {
            auto last = x_ps.rank().get_length() - 1;
            if (x_ps[last].is_static() && cos_ps[last].is_static() &&
                x_ps[last].get_length() != cos_ps[last].get_length()) {
                if (x_ps[last].get_length() == 2 * cos_ps[last].get_length()) {
                    cos_needs_dup = true;
                } else {
                    return false;
                }
            }
        }

        ov::Output<ov::Node> cos_full = cos_val;
        ov::Output<ov::Node> sin_full = sin_val;
        std::shared_ptr<ov::Node> cos_cat_node, sin_cat_node;
        if (cos_needs_dup) {
            cos_cat_node = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{cos_val, cos_val}, -1);
            sin_cat_node = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{sin_val, sin_val}, -1);
            cos_full = cos_cat_node->output(0);
            sin_full = sin_cat_node->output(0);
        }

        // Emit canonical form: split x into halves, negate second half, concat
        // to form x_rot, then compute x*cos + x_rot*sin.
        if (!x_ps.rank().is_static() ||
            !x_ps[x_ps.rank().get_length() - 1].is_static()) return false;
        int64_t half_ndims = x_ps[x_ps.rank().get_length() - 1].get_length() / 2;

        auto axis_const_new = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{}, {static_cast<int64_t>(r - 1)});
        auto split_lengths = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{2}, {half_ndims, half_ndims});
        auto new_vsplit = std::make_shared<ov::op::v1::VariadicSplit>(
            x_val, axis_const_new, split_lengths);

        auto neg_one = ov::op::v0::Constant::create(
            x_val.get_element_type(), ov::Shape{}, {-1.0f});
        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(new_vsplit->output(1), neg_one);
        auto x_rot = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{x2_neg->output(0), new_vsplit->output(0)}, -1);
        auto x_cos = std::make_shared<ov::op::v1::Multiply>(x_val, cos_full);
        auto xrot_sin = std::make_shared<ov::op::v1::Multiply>(x_rot->output(0), sin_full);
        auto new_out = std::make_shared<ov::op::v1::Add>(x_cos->output(0), xrot_sin->output(0));

        new_out->set_friendly_name(concat_node->get_friendly_name());
        ov::copy_runtime_info(
            {concat_node->get_input_node_shared_ptr(0),
             concat_node->get_input_node_shared_ptr(1),
             mul_a, mul_b, mul_c, mul_d, concat_node},
            {x2_neg, x_rot, x_cos, xrot_sin, new_out});
        ov::replace_node(concat_node, new_out);
        return true;
    };

    auto concat_pattern = wrap_type<ov::op::v0::Concat>();
    auto m = std::make_shared<Matcher>(concat_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
