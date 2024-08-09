// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_split_decomposition.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::pass::pattern;

void pass::MatmulGatherDecomposition::split_weights(const Output<Node>& weights,
                                                    OutputVector& new_weights,
                                                    Output<Node>* bias,
                                                    OutputVector& new_bias,
                                                    const bool transpos_b) {
    // weights is static
    if (weights.get_partial_shape().size() != 2u) {
        return;
    }

    if (bias) {
        const auto& bias_shape = bias->get_partial_shape();
        if (bias_shape.is_dynamic()) {
            return;
        }
        auto bias_rank = bias_shape.rank().get_length();
        if (bias_rank != 3 && bias_rank != 1) {
            return;
        }
    }

    // Decompose weights
    auto axis = register_new_node(op::v0::Constant::create(element::i32, Shape{}, {transpos_b ? 0 : 1}));
    auto split = register_new_node<op::v1::Split>(weights, axis, 3);
    for (auto& out : split->outputs()) {
        new_weights.emplace_back(out);
    }

    if (bias) {
        // Decompose bias
        auto axis2 = register_new_node(op::v0::Constant::create(element::i32, Shape{}, {-1}));  // axis -1
        auto split2 = register_new_node<op::v1::Split>(*bias, axis2, 3);
        for (auto& out : split2->outputs()) {
            new_bias.emplace_back(out);
        }
    }
}

pass::MatmulGatherDecomposition::MatmulGatherDecomposition() {
    MATCHER_SCOPE(MatmulGatherDecomposition);
    auto input_pattern = any_input();
    auto matmul_pattern = wrap_type<opset1::MatMul>({input_pattern, any_input()});

    auto bias_pattern = wrap_type<opset1::Constant>();
    auto add_pattern = wrap_type<opset1::Add>({matmul_pattern, bias_pattern});

    auto reshape_productor_pattern = std::make_shared<pattern::op::Or>(OutputVector{matmul_pattern, add_pattern});

    // Heuristics: Rank == 5, Baichun also match this pattern, but it only has rank 4, and have performance regression,
    // so filter it out.
    auto reshape_pattern =
        wrap_type<opset1::Reshape>({reshape_productor_pattern, any_input()}, ov::pass::pattern::rank_equals(5));

    // Heuristics: there should be only 3 gathers to split
    auto transpose_pattern =
        wrap_type<opset1::Transpose>({reshape_pattern, any_input()}, ov::pass::pattern::consumers_count(3));
    auto reshape2_pattern =
        wrap_type<opset1::Reshape>({reshape_pattern, any_input()}, ov::pass::pattern::consumers_count(3));

    auto reshape_or_transpose_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{reshape2_pattern, transpose_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto root_node = m.get_match_root();

        const auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();
        const auto weights = matmul->input_value(1);
        if (!weights.get_partial_shape().is_static()) {
            return false;
        }
        const std::shared_ptr<ov::Node> add =
            pattern_map.count(add_pattern) ? pattern_map.at(add_pattern).get_node_shared_ptr() : nullptr;

        const bool& transpose_b = as_type_ptr<opset1::MatMul>(matmul)->get_transpose_b();
        const auto& reshape = pattern_map.at(reshape_pattern);
        const auto reshape_input1 = reshape.get_node_shared_ptr()->input_value(1);

        NodeVector gathers, fake_quantizes;
        gathers.resize(3);
        fake_quantizes.resize(3);
        for (const auto& child : root_node->get_output_target_inputs(0)) {
            std::shared_ptr<ov::Node> fq = nullptr;
            auto gather = child.get_node()->shared_from_this();
            if (ov::is_type<opset1::FakeQuantize>(gather)) {
                fq = gather;
                gather = gather->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }
            if (ov::is_type<ov::op::util::GatherBase>(gather)) {
                const auto axis_node = as_type_ptr<opset1::Constant>(gather->get_input_node_shared_ptr(2));
                if (axis_node) {
                    const auto& axis_val = axis_node->cast_vector<int32_t>();
                    if (axis_val.size() != 1u || axis_val[0] != 0) {
                        return false;
                    }
                } else {
                    return false;
                }

                const auto indices_node = as_type_ptr<opset1::Constant>(gather->get_input_node_shared_ptr(1));
                if (indices_node) {
                    const auto& indices_val = indices_node->cast_vector<int32_t>();
                    if (indices_val.size() != 1u) {
                        return false;
                    }
                    if (indices_val[0] < 0 || indices_val[0] >= 3) {
                        return false;
                    }
                    gathers[indices_val[0]] = gather;
                    fake_quantizes[indices_val[0]] = fq;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        if (std::any_of(gathers.begin(), gathers.end(), [](const std::shared_ptr<Node> node_ptr) {
                return !node_ptr || !is_type<ov::op::util::GatherBase>(node_ptr);
            })) {
            return false;
        }

        Output<Node> bias;
        OutputVector new_weights, new_bias;
        if (add) {
            bias = pattern_map.at(bias_pattern);
        }
        split_weights(weights, new_weights, (add != nullptr) ? &bias : nullptr, new_bias, transpose_b);
        if (new_weights.size() != 3u || ((add != nullptr) && new_bias.size() != 3u)) {
            return false;
        }

        // Heuristics: Split at axis 2, new Gahter should remove it.
        const auto const_indices = register_new_node(op::v0::Constant::create(element::i32, Shape{4}, {0, 1, 3, 4}));
        const auto const_axis = register_new_node(op::v0::Constant::create(element::i32, Shape{}, {0}));
        const auto new_shape = register_new_node<op::v8::Gather>(reshape_input1, const_indices, const_axis);
        const auto& input = pattern_map.at(input_pattern);
        for (size_t i = 0; i < 3u; i++) {
            const auto new_mm = register_new_node<op::v0::MatMul>(input, new_weights[i], false, transpose_b);
            std::shared_ptr<ov::Node> reshape_productor = new_mm;
            if (add) {
                reshape_productor = register_new_node<op::v1::Add>(new_mm, new_bias[i]);
            }
            const auto new_reshape = register_new_node<op::v1::Reshape>(reshape_productor, new_shape, true);
            ov::NodeVector from_nodes = {gathers[i], weights.get_node_shared_ptr(), matmul};
            if (add) {
                from_nodes.emplace_back(add);
                from_nodes.emplace_back(pattern_map.at(bias_pattern).get_node_shared_ptr());
            }
            if (as_type<opset1::Transpose>(root_node.get()))
                from_nodes.emplace_back(root_node);

            copy_runtime_info(from_nodes, get_new_nodes());
            // Original transpose order[2,0,3,1,4], new order should be[0,2,1,3] after first axis is removed.
            const auto transpose_order =
                register_new_node(op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
            const auto new_transpose = register_new_node<op::v1::Transpose>(new_reshape, transpose_order);
            new_transpose->set_friendly_name(gathers[i]->get_friendly_name());

            if (fake_quantizes[i]) {
                fake_quantizes[i]->set_argument(0, new_transpose);
                replace_node(gathers[i], fake_quantizes[i]);
            } else {
                replace_node(gathers[i], new_transpose);
            }
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_or_transpose_pattern, matcher_name);
    this->register_matcher(m, callback);
}