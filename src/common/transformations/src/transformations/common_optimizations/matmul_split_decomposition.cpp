// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/matmul_split_decomposition.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include "openvino/op/gather.hpp"

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov;
using namespace ov::pass::pattern;

#define PRINT std::cout << "== " << __FUNCTION__ << ":" << __LINE__ << ", "

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
            PRINT << "Matched, exit";
            PRINT << "weights_rank=" << weights_rank << ", bias_rank=" << bias_rank << "\n\n";
            return;
        }
    }

    PRINT << "Matched 3.1=============\n\n";
    // Decompose weights
    auto axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));  // axis 0
    auto split = register_new_node<opset1::Split>(weights, axis, 3);
    for (auto& out : split->outputs()) {
        new_weights.emplace_back(out);
    }

    if (have_bias) {
        PRINT << "Matched 3.2 split_bias =============\n\n";
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

    auto reshape_or_transpose_pattern = std::make_shared<pattern::op::Or>(OutputVector{reshape2_pattern, transpose_pattern});

    PRINT << "Matched 1:==========\n\n";
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        PRINT << "Matched 2:==========\n\n";
        const auto& pattern_map = m.get_pattern_value_map();
        
        // Heuristics: there should be only 3 gathers to split
        auto root_node = m.get_match_root();
        bool have_transpose = as_type<opset1::Transpose>(root_node.get()) != nullptr;
        PRINT << "Matched 2.0 have_transpose = " << have_transpose << " :==========\n\n";
        auto children = root_node->get_output_target_inputs(0);
        if (children.size() != 3u) {
            PRINT << "Matched 2: EXIT: children.size() =" << children.size() << ", but expected = 3" << std::endl;
            std::cout << children.begin()->get_node()->shared_from_this()->get_type_name() << std::endl;
            return false;
        }

        auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();
        auto weights = matmul->input_value(1);
        std::shared_ptr<ov::Node> add = nullptr;
        bool have_bias = false;
        for (auto& consumer : matmul->get_output_target_inputs(0)) {
            if (ov::is_type<opset1::Add>(consumer.get_node()->shared_from_this())) {
                PRINT << "Matched 2.1 with bias :==========\n\n";
                add = pattern_map.at(add_pattern).get_node_shared_ptr();
                have_bias = true;
                break;
            }
        }
        PRINT << "Matched 2.1: matmul=" << matmul->get_friendly_name() << "\n\n";

        const auto& reshape = pattern_map.at(reshape_pattern);
        auto concat = reshape.get_node_shared_ptr()->input_value(1);

        NodeVector gathers;
        gathers.resize(3);
        for (auto& child : children) {
            auto gather = child.get_node()->shared_from_this();
            if (ov::is_type<ov::op::util::GatherBase>(gather)) {
                const auto axis_node = as_type_ptr<opset6::Constant>(gather->input_value(2).get_node_shared_ptr());
                if (axis_node) {
                    const auto& axis_val = axis_node->cast_vector<int32_t>();
                    if (axis_val.size() != 1u || axis_val[0] != 0) {
                        PRINT << "Matched 2: EXIT: axis_val.size() != 1 || axis_val[0] != 0" << std::endl;
                        PRINT << "Matched 2: EXIT: axis_val.size()=" << axis_val.size()
                              << ", axis_val[0]=" << axis_val[0] << std::endl;
                        return false;
                    }
                } else {
                    PRINT << "Matched 2: EXIT: axis_node is not Constant" << std::endl;
                    return false;
                }

                const auto indices_node = as_type_ptr<opset6::Constant>(gather->input_value(1).get_node_shared_ptr());
                if (indices_node) {
                    const auto& indices_val = indices_node->cast_vector<int32_t>();
                    if (indices_val.size() != 1) {
                        PRINT << "Matched 2: EXIT:indices_val.size()=" << indices_val.size() << std::endl;
                        return false;
                    }
                    if (indices_val[0] < 0 || indices_val[0] >= 3) {
                        PRINT << "Matched 2: EXIT:indices_val[0]=" << indices_val[0] << std::endl;
                        return false;
                    }
                    gathers[indices_val[0]] = gather;
                } else {
                    PRINT << "Matched 2: EXIT:indices_node is not Constant" << std::endl;
                    return false;
                }
            } else {
                PRINT << "Matched 2: EXIT:child is not gather\n\n";
                return false;
            }
        }

        if (std::any_of(gathers.begin(), gathers.end(), [](const std::shared_ptr<Node> node_ptr) {
                return !node_ptr || !is_type<ov::op::util::GatherBase>(node_ptr);
            })) {
            PRINT << "Matched 2: EXIT:Not all consumer are gather\n\n";
            return false;
        }

        PRINT << "Matched 3: split_weights====================\n\n";
        OutputVector new_weights, new_bias;
        split_weights(weights,
                      new_weights,
                      have_bias,
                      have_bias ? pattern_map.at(bias_pattern) : Output<Node>(),
                      new_bias);
        if (new_weights.size() != 3u || (have_bias && new_bias.size() != 3u)) {
            PRINT << "Matched 3: Exit\n\n";
            PRINT << "new_weights.size()=" << new_weights.size() << ", have_bias=" << have_bias << "\n";
            if (have_bias) {
                PRINT << "new_bias.size()=" << new_bias.size() << std::endl;
            }
            return false;
        }

        auto const_indices = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 1, 3, 4}));
        auto const_axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto new_shape = register_new_node<v1::Gather>(concat, const_indices, const_axis);

        PRINT << "Matched 4: replace ===================\n\n";
        const auto& input = pattern_map.at(input_pattern);
        for (size_t i = 0; i < 3u; i++) {
            PRINT << "Matched 4.1: replace " << i << "=============\n";
            auto mm0 = register_new_node<v0::MatMul>(input, new_weights[i], false, true);
            PRINT << "Matched 4.2: replace " << i << "=============\n";
            std::shared_ptr<ov::Node> reshape_productor = mm0;
            PRINT << "Matched 4.3: replace " << i << "=============\n";
            if (have_bias) {
                reshape_productor = register_new_node<v1::Add>(mm0, new_bias[i]);
            }
            auto reshape0 = register_new_node<v1::Reshape>(reshape_productor, new_shape, true);
            PRINT << "Matched 4.4: replace " << i << "=============\n";
            ov::NodeVector from_nodes = {gathers[i],
                                         weights.get_node_shared_ptr(),
                                         matmul};
            if (have_bias) {
                from_nodes.emplace_back(add);
                from_nodes.emplace_back(pattern_map.at(bias_pattern).get_node_shared_ptr());
            }
            if (have_transpose)
                from_nodes.emplace_back(root_node);

            copy_runtime_info(from_nodes, get_new_nodes());
            auto transpose_order = register_new_node(v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
            auto transpose0 = register_new_node<v1::Transpose>(reshape0, transpose_order);
            transpose0->set_friendly_name(gathers[i]->get_friendly_name());

            PRINT << "Matched 4.5: replace transpose =============\n";
            replace_node(gathers[i], transpose0);  // replace gatherX by transposeX

            PRINT << "Matched 4: replace " << i << " done =========\n\n";
        }

        PRINT << "Matched 5: Absolutely matched finish.:==========\n\n";
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_or_transpose_pattern, matcher_name);
    this->register_matcher(m, callback);
}

void pass::MatmulVariadicSplitDecomposition::split_weights(const Output<Node>& weights,
                                                           const bool& transpose_b,
                                                           OutputVector& new_weights,
                                                           const Output<Node>& split_length) {
    const auto& weights_shape = weights.get_partial_shape();
    int64_t weights_rank = static_cast<int64_t>(weights_shape.rank().get_length());

    const auto& split_length_shape = split_length.get_partial_shape();
    int64_t split_length_rank = static_cast<int64_t>(split_length_shape.rank().get_length());

    // const auto& bias_shape = bias.get_partial_shape();
    // int64_t bias_rank = static_cast<int64_t>(bias_shape.rank().get_length());
    PRINT << "weights_rank=" << weights_rank << ", weights_shape=" << weights_shape << "\n\n";
    PRINT << "split_length_rank=" << split_length_rank << ", split_length_shape=" << split_length_shape << "\n\n";
    if (weights_rank != 2) {
        PRINT << "weights_rank != 2, return." << "\n\n";
        return;
    }

    PRINT << " 1=============\n\n";
    auto axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {transpose_b ? 0 : 1}));  // axis 0
    auto variadic_split = register_new_node<opset13::VariadicSplit>(weights, axis, split_length);

    // Constantfold new weights
    for (auto& out : variadic_split->outputs()) {
        if (auto constant = ov::util::get_constant_from_source(out)) {  // TODO: why Convert cannot be constfolded?
            new_weights.emplace_back(constant->shared_from_this());
        } else
            new_weights.emplace_back(out);
    }
}

pass::MatmulVariadicSplitDecomposition::MatmulVariadicSplitDecomposition() {
    MATCHER_SCOPE(MatmulVariadicSplitDecomposition);
    auto input_pattern = any_input();
    auto matmul_pattern = wrap_type<opset1::MatMul>({input_pattern, any_input()});
    auto variadic_split_pattern =
        wrap_type<opset13::VariadicSplit>({matmul_pattern, wrap_type<v0::Constant>(), wrap_type<v0::Constant>()});

    PRINT << "Matched 1: =======================\n\n";
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        PRINT << "Matched 2: callback=======================\n\n";
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul = pattern_map.at(matmul_pattern).get_node_shared_ptr();
        auto weights = matmul->input_value(1);
        auto variadic_split = pattern_map.at(variadic_split_pattern).get_node_shared_ptr();
        auto split_length = variadic_split->input_value(2);

        // Heuristics: MatMul transpose_a==false
        auto mm_ptr = ov::as_type_ptr<opset1::MatMul>(matmul);
        if (!mm_ptr) {
            return false;
        }
        if (mm_ptr->get_transpose_a() != false) {
            PRINT << "Matched 1: Exit. mm_ptr->get_transpose_a() != false FAIL\n\n";
            return false;
        }
        auto transpose_b = mm_ptr->get_transpose_b();

        // Heuristics: Must be split into 3 nodes.
        if (variadic_split->get_output_size() != 3u) {
            PRINT << "variadic_split->get_output_size()[" << variadic_split->get_output_size()
                   << "] != 3u ==========================\n\n";
            return false;
        }
        // axis = matmal output shape size - 1
        const auto axis_node = ov::as_type_ptr<opset6::Constant>(variadic_split->input_value(1).get_node_shared_ptr());
        if (axis_node) {
            const auto& axis_val = axis_node->cast_vector<int32_t>();
            if (axis_val.size() != 1u || static_cast<size_t>(axis_val[0]) != matmul->get_output_shape(0).size() - 1u) {
                PRINT << "axis_val.size() != 1u || axis_val[0] != " << matmul->get_output_shape(0).size() << std::endl;
                PRINT << "axis_val.size() = " << axis_val.size() << std::endl;
                PRINT << "axis_val[0] = " << axis_val[0] << std::endl;
                return false;
            }
        }
        else {
            return false;
        }

        // The consumer are 3 Reshapes
        std::vector<ov::Input<ov::Node>> reshapes;
        for (size_t i = 0; i < 3u; i++) {
            auto variadic_splic_consumers = variadic_split->get_output_target_inputs(i);
            for (auto consumer : variadic_splic_consumers) {
                if (ov::is_type<opset1::Reshape>(consumer.get_node())) {
                    reshapes.push_back(consumer);
                } else {
                    PRINT << "variadic_splic_consumers is not Reshape." << std::endl;
                    PRINT << "consumer.get_node() type=" << consumer.get_node()->get_type_name() << std::endl;
                    return false;
                }
            }
        }

        OutputVector new_weights;
        split_weights(weights, transpose_b, new_weights, split_length);
        if (new_weights.size() != 3u || reshapes.size() != 3u) {
            PRINT << "new_weights.size(),reshapes.size() = " << new_weights.size() << ", " << reshapes.size()
                   << std::endl;
            return false;
        }

        // Replace MatMul+Split with 3 MatMul
        const auto& input = pattern_map.at(input_pattern);
        for (size_t i = 0 ; i < 3u; i++) {
            PRINT << "Replace MatMul:" << i << std::endl; 
            auto mm_new = register_new_node<v0::MatMul>(input, new_weights[i], false, transpose_b);
            reshapes[i].get_node()->set_argument(0, mm_new->output(0));
            PRINT << "Replace MatMul:" << i << ", Done" << std::endl;
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(variadic_split_pattern, matcher_name);
    this->register_matcher(m, callback);
}