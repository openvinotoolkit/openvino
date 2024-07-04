// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_split_decomposition.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov;
using namespace ov::pass::pattern;

void pass::MatmulGatherDecomposition::split_weights(const Output<Node>& weights,
                                                    OutputVector& new_weights,
                                                    const bool& have_bias,
                                                    const Output<Node>& bias,
                                                    OutputVector& new_bias) {
    const auto& weights_shape = weights.get_partial_shape();
    int64_t weights_rank = static_cast<int64_t>(weights_shape.rank().get_length());

    if (have_bias) {
        const auto& bias_shape = bias.get_partial_shape();
        int64_t bias_rank = static_cast<int64_t>(bias_shape.rank().get_length());
        if (weights_rank != 2 || (bias_rank != 3 && bias_rank != 1)) {
            return;
        }
    }

    // Decompose weights
    auto axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));  // axis 0
    auto split = register_new_node<opset1::Split>(weights, axis, 3);
    for (auto& out : split->outputs()) {
        new_weights.emplace_back(out);
    }

    if (have_bias) {
        // Decompose bias
        auto axis2 = register_new_node(v0::Constant::create(element::i32, Shape{}, {-1}));  // axis -1
        auto split2 = register_new_node<opset1::Split>(bias, axis2, 3);
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

    auto reshape_pattern = wrap_type<opset1::Reshape>({reshape_productor_pattern, any_input()});
    auto transpose_pattern = wrap_type<opset6::Transpose>({reshape_pattern, any_input()});
    auto reshape2_pattern = wrap_type<opset1::Reshape>({reshape_pattern, any_input()});

    auto reshape_or_transpose_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{reshape2_pattern, transpose_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        // Heuristics: there should be only 3 gathers to split
        auto root_node = m.get_match_root();
        bool have_transpose = as_type<opset1::Transpose>(root_node.get()) != nullptr;
        auto children = root_node->get_output_target_inputs(0);
        if (children.size() != 3u) {
            return false;
        }

        auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();
        auto weights = matmul->input_value(1);
        std::shared_ptr<ov::Node> add = nullptr;
        bool have_bias = false;
        for (auto& consumer : matmul->get_output_target_inputs(0)) {
            if (ov::is_type<opset1::Add>(consumer.get_node()->shared_from_this())) {
                add = pattern_map.at(add_pattern).get_node_shared_ptr();
                have_bias = true;
                break;
            }
        }
        const auto& reshape = pattern_map.at(reshape_pattern);
        auto concat = reshape.get_node_shared_ptr()->input_value(1);

        NodeVector gathers, fake_quantizes;
        gathers.resize(3);
        fake_quantizes.resize(3);
        for (auto& child : children) {
            std::shared_ptr<ov::Node> fq = nullptr;
            auto gather = child.get_node()->shared_from_this();
            if (ov::is_type<opset1::FakeQuantize>(gather)) {
                fq = gather;
                gather = gather->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }
            if (ov::is_type<ov::op::util::GatherBase>(gather)) {
                const auto axis_node = as_type_ptr<opset6::Constant>(gather->input_value(2).get_node_shared_ptr());
                if (axis_node) {
                    const auto& axis_val = axis_node->cast_vector<int32_t>();
                    if (axis_val.size() != 1u || axis_val[0] != 0) {
                        return false;
                    }
                } else {
                    return false;
                }

                const auto indices_node = as_type_ptr<opset6::Constant>(gather->input_value(1).get_node_shared_ptr());
                if (indices_node) {
                    const auto& indices_val = indices_node->cast_vector<int32_t>();
                    if (indices_val.size() != 1) {
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

        OutputVector new_weights, new_bias;
        split_weights(weights,
                      new_weights,
                      have_bias,
                      have_bias ? pattern_map.at(bias_pattern) : Output<Node>(),
                      new_bias);
        if (new_weights.size() != 3u || (have_bias && new_bias.size() != 3u)) {
            return false;
        }

        auto const_indices = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 1, 3, 4}));
        auto const_axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto new_shape = register_new_node<v1::Gather>(concat, const_indices, const_axis);
        const auto& input = pattern_map.at(input_pattern);
        for (size_t i = 0; i < 3u; i++) {
            auto new_mm = register_new_node<v0::MatMul>(input, new_weights[i], false, true);
            std::shared_ptr<ov::Node> reshape_productor = new_mm;
            if (have_bias) {
                reshape_productor = register_new_node<v1::Add>(new_mm, new_bias[i]);
            }
            auto new_reshape = register_new_node<v1::Reshape>(reshape_productor, new_shape, true);
            ov::NodeVector from_nodes = {gathers[i], weights.get_node_shared_ptr(), matmul};
            if (have_bias) {
                from_nodes.emplace_back(add);
                from_nodes.emplace_back(pattern_map.at(bias_pattern).get_node_shared_ptr());
            }
            if (have_transpose)
                from_nodes.emplace_back(root_node);

            copy_runtime_info(from_nodes, get_new_nodes());
            auto transpose_order = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
            auto new_transpose = register_new_node<v1::Transpose>(new_reshape, transpose_order);
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