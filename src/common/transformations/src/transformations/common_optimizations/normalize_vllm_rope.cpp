// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_vllm_rope.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

bool is_last_axis(int64_t axis, const ov::Rank& rank) {
    return rank.is_static() && (axis == -1 || axis == rank.get_length() - 1);
}

}  // namespace

ov::pass::NormalizeVLLMRoPE::NormalizeVLLMRoPE() {
    MATCHER_SCOPE(NormalizeVLLMRoPE);
    using namespace ov::pass::pattern;
    using ov::pass::operator|;
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;

    // The split's axis is a Constant naming the last dimension of the split input.
    const auto splits_last_axis = op::Predicate(
        [](const Output<Node>& output) {
            const auto node = output.get_node();
            const auto axis = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
            return axis && ov::shape_size(axis->get_shape()) == 1 &&
                   is_last_axis(axis->cast_vector<int64_t>()[0], node->get_input_partial_shape(0).rank());
        },
        "splits_last_axis");
    const auto concats_last_axis = op::Predicate(
        [](const Output<Node>& output) {
            const auto concat = ov::as_type_ptr<v0::Concat>(output.get_node_shared_ptr());
            return concat && is_last_axis(concat->get_axis(), output.get_partial_shape().rank());
        },
        "concats_last_axis");
    // Both halves, cos and sin end in the same static or symbolic width HALF.
    const auto half_width = shape_matches("..., HALF");

    auto x_m = any_input(has_static_rank());
    auto split_m =
        wrap_type_strict_index<v1::Split>({x_m, wrap_const()},
                                          splits_last_axis && half_width && attrs_match({{"num_splits", 2}}));
    auto variadic_split_m =
        wrap_type_strict_index<v1::VariadicSplit>({x_m, wrap_const(), wrap_const()}, splits_last_axis && half_width);
    auto x1_m = split_m->output(0) | variadic_split_m->output(0);
    auto x2_m = split_m->output(1) | variadic_split_m->output(1);
    auto cos_m = any_input(half_width);
    auto sin_m = any_input(half_width);

    // Concat(x1 * cos - x2 * sin, x2 * cos + x1 * sin), with the subtraction as Multiply(-1) + Add
    auto x1_cos_m = wrap_type<v1::Multiply>({x1_m, cos_m});
    auto x2_sin_m = wrap_type<v1::Multiply>({x2_m, sin_m});
    auto neg_x2_sin_m = wrap_type<v1::Multiply>({x2_sin_m, -1.0f});
    auto first_half_m = wrap_type<v1::Add>({x1_cos_m, neg_x2_sin_m});
    auto x2_cos_m = wrap_type<v1::Multiply>({x2_m, cos_m});
    auto x1_sin_m = wrap_type<v1::Multiply>({x1_m, sin_m});
    auto second_half_m = wrap_type<v1::Add>({x2_cos_m, x1_sin_m});
    auto concat_m = wrap_type<v0::Concat>({first_half_m, second_half_m}, concats_last_axis);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& x = pattern_map.at(x_m);
        const auto& cos = pattern_map.at(cos_m);
        const auto& sin = pattern_map.at(sin_m);
        // The fused RoPE reads cos and sin as one table shape; the token dims are dynamic, so this can't be a symbol.
        if (cos.get_partial_shape() != sin.get_partial_shape()) {
            return false;
        }
        // The VariadicSplit lengths below need a static width.
        const auto symbols = m.get_symbols();
        const auto& half = symbols.at("HALF");
        if (!half.is_integer()) {
            return false;
        }

        auto cos_full = std::make_shared<v0::Concat>(OutputVector{cos, cos}, -1);
        auto sin_full = std::make_shared<v0::Concat>(OutputVector{sin, sin}, -1);

        const auto last_axis = static_cast<int64_t>(x.get_partial_shape().size()) - 1;
        auto axis = v0::Constant::create(element::i64, Shape{}, {last_axis});
        auto lengths = v0::Constant::create(element::i64, Shape{2}, {half.i(), half.i()});
        auto split = std::make_shared<v1::VariadicSplit>(x, axis, lengths);
        auto minus_one = v0::Constant::create(x.get_element_type(), Shape{}, {-1});
        auto neg_x2 = std::make_shared<v1::Multiply>(split->output(1), minus_one);
        auto rotated = std::make_shared<v0::Concat>(OutputVector{neg_x2, split->output(0)}, -1);
        auto x_cos = std::make_shared<v1::Multiply>(x, cos_full);
        auto rotated_sin = std::make_shared<v1::Multiply>(rotated, sin_full);
        auto rope = std::make_shared<v1::Add>(x_cos, rotated_sin);
        const NodeVector
            new_nodes{cos_full, sin_full, axis, lengths, split, minus_one, neg_x2, rotated, x_cos, rotated_sin, rope};

        const auto concat = pattern_map.at(concat_m).get_node_shared_ptr();
        // Multi-output pattern nodes are read from the node map.
        const auto node_map = m.get_pattern_map();
        NodeVector replaced{node_map.count(split_m) ? node_map.at(split_m) : node_map.at(variadic_split_m)};
        for (const auto& label : {x1_cos_m, x2_sin_m, neg_x2_sin_m, first_half_m, x2_cos_m, x1_sin_m, second_half_m}) {
            replaced.push_back(pattern_map.at(label).get_node_shared_ptr());
        }
        replaced.push_back(concat);
        ov::copy_runtime_info(replaced, new_nodes);
        rope->set_friendly_name(concat->get_friendly_name());
        ov::replace_node(concat, rope);
        return true;
    };

    auto m = std::make_shared<Matcher>(concat_m, matcher_name);
    register_matcher(m, callback);
}
