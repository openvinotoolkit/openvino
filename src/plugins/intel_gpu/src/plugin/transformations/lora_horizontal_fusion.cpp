// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_horizontal_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"

namespace ov::intel_gpu {

LoRAHorizontalFusion::LoRAHorizontalFusion() {
    using namespace ov::pass::pattern;

    auto is_target_pattern = [](const std::shared_ptr<Node>& split_node) {
        auto is_lora_pattern = [](const std::shared_ptr<Node>& node) {
            #define check(node) if (!node) return false;

            const auto& add = ov::as_type_ptr<ov::op::v1::Add>(node);                                                         check(add)

            size_t matmul2_idx = ov::is_type<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(0)) ? 0 : 1;
            const auto& matmul2 = ov::as_type_ptr<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(matmul2_idx));           check(matmul2)

            const auto& multiply = ov::as_type_ptr<ov::op::v1::Multiply>(matmul2->get_input_node_shared_ptr(0));              check(multiply)

            const auto& variable_b = ov::as_type_ptr<ov::op::util::ReadValueBase>(matmul2->get_input_node_shared_ptr(1));     check(variable_b)

            size_t matmul1_idx = ov::is_type<ov::op::v0::MatMul>(multiply->get_input_node_shared_ptr(0)) ? 0 : 1;
            const auto& matmul1 = ov::as_type_ptr<ov::op::v0::MatMul>(multiply->get_input_node_shared_ptr(matmul1_idx));      check(matmul1)

            size_t alpha_idx = (matmul1_idx + 1) % 2;
            const auto& variable_alpha =
                ov::as_type_ptr<ov::op::util::ReadValueBase>(multiply->get_input_node_shared_ptr(alpha_idx));                 check(variable_alpha)

            const auto& variable_a = ov::as_type_ptr<ov::op::util::ReadValueBase>(matmul1->get_input_node_shared_ptr(1));     check(variable_a)

            #undef check
            return true;
        };

        for (const auto& user : split_node->get_users()) {
            if (!is_lora_pattern(user)) {
                return false;
            }
        }

        return true;
    };

    auto lora_input = any_input();
    auto main_flow_1 = wrap_type<op::FullyConnectedCompressed>({lora_input, any_input(), any_input(), any_input()});
    auto main_flow_2 = wrap_type<op::FullyConnectedCompressed>({lora_input, any_input(), any_input(), any_input(), any_input()});
    auto main_flow = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{main_flow_1, main_flow_2});

    auto axis_const = wrap_type<ov::op::v0::Constant>();
    auto split_const = wrap_type<ov::op::v0::Constant>();
    auto split = wrap_type<ov::op::v1::VariadicSplit>({main_flow, axis_const, split_const}, is_target_pattern);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& split = m.get_match_root();

        ov::NodeVector add_nodes;
        ov::NodeVector multiply_nodes;
        ov::NodeVector variable_a_nodes;
        ov::NodeVector variable_b_nodes;
        ov::NodeVector variable_alpha_nodes;
        ov::NodeVector matmul1_nodes;
        ov::NodeVector matmul2_nodes;

        for (const auto& add : split->get_users()) {
            add_nodes.emplace_back(add);

            size_t matmul2_idx = ov::is_type<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(0)) ? 0 : 1;
            matmul2_nodes.emplace_back(add->get_input_node_shared_ptr(matmul2_idx));
        }
        for (const auto& matmul2 : matmul2_nodes) {
            multiply_nodes.emplace_back(matmul2->get_input_node_shared_ptr(0));
            variable_b_nodes.emplace_back(matmul2->get_input_node_shared_ptr(1));
        }
        for (const auto& multiply : multiply_nodes) {
            size_t matmul1_idx = ov::is_type<ov::op::v0::MatMul>(multiply->get_input_node_shared_ptr(0)) ? 0 : 1;
            matmul1_nodes.emplace_back(multiply->get_input_node_shared_ptr(matmul1_idx));

            size_t alpha_idx = (matmul1_idx + 1) % 2;
            variable_alpha_nodes.emplace_back(multiply->get_input_node_shared_ptr(alpha_idx));
        }
        for (const auto& matmul1 : matmul1_nodes) {
            variable_a_nodes.emplace_back(matmul1->get_input_node_shared_ptr(1));
        }

        auto fused_variable_a = std::make_shared<ov::op::v0::Concat>(variable_a_nodes, 0);
        fused_variable_a->set_friendly_name(variable_a_nodes[0]->get_friendly_name() +
                                            "_fused_" + std::to_string(variable_a_nodes.size()) + "_ReadValues");
        ov::copy_runtime_info(variable_a_nodes, fused_variable_a);

        auto fused_variable_alpha = std::make_shared<ov::op::v0::Concat>(variable_alpha_nodes, 1);
        fused_variable_alpha->set_friendly_name(variable_alpha_nodes[0]->get_friendly_name() +
                                                "_fused_" + std::to_string(variable_alpha_nodes.size()) + "_ReadValues");
        ov::copy_runtime_info(variable_alpha_nodes, fused_variable_alpha);

        bool transpose_a1 = ov::as_type_ptr<ov::op::v0::MatMul>(matmul1_nodes[0])->get_transpose_a();
        bool transpose_b1 = ov::as_type_ptr<ov::op::v0::MatMul>(matmul1_nodes[0])->get_transpose_b();
        auto fused_matmul1 = std::make_shared<ov::op::v0::MatMul>(pattern_map.at(lora_input), fused_variable_a, transpose_a1, transpose_b1);
        auto fused_matmul1_name = matmul1_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(matmul1_nodes.size()) + "_MatMuls";
        fused_matmul1->set_friendly_name(fused_matmul1_name);
        ov::copy_runtime_info(matmul1_nodes, fused_matmul1);
        for (const auto& old_matmul1 : matmul1_nodes) {
            old_matmul1->clear_control_dependencies();
        }

        auto fused_multiply = std::make_shared<ov::op::v1::Multiply>(fused_matmul1, fused_variable_alpha);
        auto multiply_name = multiply_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(multiply_nodes.size()) + "_Multiply";
        fused_multiply->set_friendly_name(multiply_name);
        ov::copy_runtime_info(multiply_nodes, fused_multiply);
        for (const auto& old_multiply : multiply_nodes) {
            old_multiply->clear_control_dependencies();
        }

        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {fused_multiply->get_output_partial_shape(0).size() - 1});
        auto output_split = std::make_shared<ov::op::v1::Split>(fused_multiply, axis_const, matmul2_nodes.size());
        auto split_name = fused_multiply->get_friendly_name() + "_split";
        copy_runtime_info(fused_multiply, output_split);
        output_split->set_friendly_name(split_name);
        for (size_t i = 0; i < matmul2_nodes.size(); ++i) {
            matmul2_nodes[i]->input(0).replace_source_output(output_split->output(i));
        }

        auto fused_matmul2 = std::make_shared<ov::op::v0::Concat>(matmul2_nodes, matmul2_nodes[0]->get_output_partial_shape(0).size() - 1);
        auto matmul2_name = matmul2_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(matmul2_nodes.size()) + "_MatMuls_output";
        fused_matmul2->set_friendly_name(matmul2_name);
        ov::copy_runtime_info(matmul2_nodes, fused_matmul2);

        auto fused_add = std::make_shared<ov::op::v1::Add>(split->get_input_node_shared_ptr(0), fused_matmul2);
        auto fused_add_name = add_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(add_nodes.size()) + "_Adds";
        fused_add->set_friendly_name(fused_add_name);
        ov::copy_runtime_info(add_nodes, fused_add);

        for (size_t i = 0; i < add_nodes.size(); ++i) {
            const auto& old_add = add_nodes[i];
            for (auto u : old_add->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == old_add) {
                        u->input(idx).replace_source_output(split->output(i));
                    }
                }
            }
            old_add->clear_control_dependencies();
        }

        split->input(0).replace_source_output(fused_add->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(split, "LoRAHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
