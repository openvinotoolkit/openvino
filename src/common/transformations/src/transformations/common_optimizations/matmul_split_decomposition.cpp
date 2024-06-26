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

    auto reshape_or_transpose_pattern =
        std::make_shared<pattern::op::Or>(OutputVector{reshape2_pattern, transpose_pattern});

    PRINT << "Matched 1:==========\n\n";
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        PRINT << "Matched 2: callback==========\n\n";
        const auto& pattern_map = m.get_pattern_value_map();

        // Heuristics: there should be only 3 gathers to split
        auto root_node = m.get_match_root();
        bool have_transpose = as_type<opset1::Transpose>(root_node.get()) != nullptr;
        auto children = root_node->get_output_target_inputs(0);
        if (children.size() != 3u) {
            PRINT << "Matched 2: EXIT: children.size() =" << children.size() << ", but expected = 3" << std::endl;
            PRINT << "Matched 2: EXIT: root type=" << children.begin()->get_node()->shared_from_this()->get_type_name()
                  << std::endl;
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
        const auto& reshape = pattern_map.at(reshape_pattern);
        auto concat = reshape.get_node_shared_ptr()->input_value(1);

        NodeVector gathers, fake_quantizes;
        gathers.resize(3);
        fake_quantizes.resize(3);
        for (auto& child : children) {
            std::shared_ptr<ov::Node> fq = nullptr;
            auto gather = child.get_node()->shared_from_this();
            if (ov::is_type<opset1::FakeQuantize>(gather)) {
                PRINT << "Have FQ before gather===========" << std::endl;
                fq = gather;
                gather = gather->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }
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
                    fake_quantizes[indices_val[0]] = fq;
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
            PRINT << "Matched 4." << i << ": replace =============\n";
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

            PRINT << "Matched 4." << i << ": replace done=============\n";
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

    if (weights_rank != 2) {
        PRINT << "Merged exit: weights_rank=" << weights_rank << ", but expected = 2, return." << "\n\n";
        PRINT << "split_length_rank=" << split_length_rank << ", split_length_shape=" << split_length_shape << "\n\n";
        return;
    }

    auto axis = register_new_node(v0::Constant::create(element::i32, Shape{}, {transpose_b ? 0 : 1}));  // axis 0
    auto variadic_split = register_new_node<opset13::VariadicSplit>(weights, axis, split_length);
    for (auto& out : variadic_split->outputs()) {
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
            PRINT << "Matched 2: Exit. =======================\n\n";
            return false;
        }
        if (mm_ptr->get_transpose_a() != false) {
            PRINT << "Matched 2: Exit. mm_ptr->get_transpose_a() != false FAIL\n\n";
            return false;
        }
        auto transpose_b = mm_ptr->get_transpose_b();

        // Heuristics: Must be split into 3 nodes.
        if (variadic_split->get_output_size() != 3u) {
            PRINT << "Matched 2: Exit. variadic_split->get_output_size()[" << variadic_split->get_output_size()
                  << "] != 3u ==========================\n\n";
            return false;
        }
        // axis = matmal output shape size - 1
        const auto axis_node = ov::as_type_ptr<opset6::Constant>(variadic_split->input_value(1).get_node_shared_ptr());
        if (axis_node) {
            const auto& axis_val = axis_node->cast_vector<int32_t>();
            const auto& mm_rank = matmul->get_output_partial_shape(0).rank().get_length();
            if (axis_val.size() != 1u || axis_val[0] != mm_rank - 1) {
                PRINT << "Matched 2: Exit. axis_val.size() != 1u || axis_val[0] != " << mm_rank << std::endl;
                PRINT << "Matched 2: Exit. axis_val.size() = " << axis_val.size() << std::endl;
                PRINT << "Matched 2: Exit. axis_val[0] = " << axis_val[0] << std::endl;
                return false;
            }
        } else {
            PRINT << "Matched 2: Exit. axis_node is not const" << std::endl;
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
                    PRINT << "Matched 2: Exit. variadic_splic_consumers is not Reshape." << std::endl;
                    PRINT << "Matched 2: Exit. consumer.get_node() type=" << consumer.get_node()->get_type_name()
                          << std::endl;
                    return false;
                }
            }
        }

        PRINT << "Matched 3: split_weights==================\n\n" << std::endl;
        OutputVector new_weights;
        split_weights(weights, transpose_b, new_weights, split_length);
        if (new_weights.size() != 3u || reshapes.size() != 3u) {
            PRINT << "Matched 3: Exit. new_weights.size(),reshapes.size() = " << new_weights.size() << ", "
                  << reshapes.size() << std::endl;
            return false;
        }

        // Replace MatMul+Split with 3 MatMul
        const auto& input = pattern_map.at(input_pattern);
        for (size_t i = 0; i < 3u; i++) {
            PRINT << "Matched 4." << i << ", Replace MatMul:" << std::endl;
            auto mm_new = register_new_node<v0::MatMul>(input, new_weights[i], false, transpose_b);
            reshapes[i].get_node()->set_argument(0, mm_new->output(0));
            copy_runtime_info({weights.get_node_shared_ptr(), matmul}, get_new_nodes());
            PRINT << "Matched 4." << i << ", Replace MatMul done" << std::endl;
        }

        PRINT << "Matched 5: Absolutely matched finish.:==========\n\n";
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(variadic_split_pattern, matcher_name);
    this->register_matcher(m, callback);
}