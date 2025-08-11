// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_subgraph_horizontal_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/matmul.hpp"

#include "ov_ops/lora_subgraph.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/lora_subgraph_fused.hpp"

namespace ov::intel_gpu {

LoRASubgraphHorizontalFusion::LoRASubgraphHorizontalFusion() {
    using namespace ov::pass::pattern;

    auto is_target_pattern = [](const std::shared_ptr<Node>& split_node) {
        for (const auto& user : split_node->get_users()) {
            if (!ov::is_type<ov::op::internal::LoraSubgraph>(user)) {
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
    auto split = wrap_type<ov::op::v1::VariadicSplit>({main_flow, axis_const, split_const}, ov::pass::pattern::op::as_value_predicate(is_target_pattern));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& split = m.get_match_root();

        ov::NodeVector lora_nodes = split->get_users();

        ov::OutputVector states;
        for (const auto& lora : lora_nodes) {
            states.emplace_back(lora->get_input_node_shared_ptr(2));
            states.emplace_back(lora->get_input_node_shared_ptr(3));
            states.emplace_back(lora->get_input_node_shared_ptr(4));
        }

        bool transposed_states = true;
        // Assumption that all states in all LoRA's are simultaneously transposed or not transposed
        const auto& any_lora = ov::as_type_ptr<ov::op::internal::LoraSubgraph>(lora_nodes[0]);
        const auto& subgraph_ops = any_lora->get_function()->get_ops();
        for (const auto& op : subgraph_ops) {
            if (ov::is_type<const ov::op::v0::MatMul>(op.get())) {
                const auto& matmul = ov::as_type<const ov::op::v0::MatMul>(op.get());
                transposed_states = matmul->get_transpose_b();
                break;
            }
        }

        auto fused_lora = std::make_shared<op::LoraSubgraphFused>(pattern_map.at(main_flow),
                                                                  pattern_map.at(lora_input),
                                                                  states,
                                                                  transposed_states);

        auto fused_lora_name = lora_nodes[0]->get_friendly_name() + "_fused_" + std::to_string(lora_nodes.size()) + "_LoRA";
        fused_lora->set_friendly_name(fused_lora_name);
        ov::copy_runtime_info(lora_nodes, fused_lora);

        for (size_t i = 0; i < lora_nodes.size(); ++i) {
            const auto& lora = lora_nodes[i];
            for (auto u : lora->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == lora) {
                        u->input(idx).replace_source_output(split->output(i));
                    }
                }
            }
            lora->clear_control_dependencies();
        }

        split->input(0).replace_source_output(fused_lora->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(split, "LoRASubgraphHorizontalFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
