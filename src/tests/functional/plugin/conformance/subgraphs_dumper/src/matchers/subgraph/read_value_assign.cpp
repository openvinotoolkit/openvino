// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "matchers/subgraph/read_value_assign.hpp"
#include "utils/model.hpp"

using namespace ov::tools::subgraph_dumper;

std::vector<ReadValueAssignExtractor::ExtractedPattern>
ReadValueAssignExtractor::extract(const std::shared_ptr<ov::Model> &model) {
    struct PairChecker {
        int cnt_assign = 0;
        int cnt_read_val = 0;
        std::shared_ptr<ov::Node> rv;
        std::string variable_id;
    };
    std::map<ov::op::util::Variable*, PairChecker>  pairs;
    for (auto& node : model->get_ordered_ops()) {
        if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::AssignBase>(node)) {
            pairs[assign->get_variable().get()].cnt_assign++;
            pairs[assign->get_variable().get()].variable_id = assign->get_variable()->get_info().variable_id;
        } else if (const auto& read_value = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(node)) {
            pairs[read_value->get_variable().get()].cnt_read_val++;
            pairs[read_value->get_variable().get()].rv = read_value;
            pairs[read_value->get_variable().get()].variable_id = read_value->get_variable()->get_info().variable_id;
        }
    }

    std::vector<ReadValueAssignExtractor::ExtractedPattern> matched_patterns;
    for (auto& pair : pairs) {
        if (pair.second.cnt_assign != 1 || pair.second.cnt_read_val != 1) {
            std::cout << "[ WARNING ] Model is incorrect. Assign and ReadValue operations must be in pairs. ";
            std::cout << "Check operations with id: " << pair.second.variable_id << std::endl;
            continue;
        }
        // node, amount of analyzed outputs, amount of analyzed target inputs of output
        std::vector<std::tuple<std::shared_ptr<ov::Node>, size_t, size_t>> bfs_queue;
        bfs_queue.push_back({pair.second.rv, 0, 0});

        std::vector<std::shared_ptr<ov::Node>> all_extracted_nodes;

        while (bfs_queue.size() != 0) {
            auto& node_element = bfs_queue.front();
            auto node = std::get<0>(node_element);
            all_extracted_nodes.push_back(node);
            if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::AssignBase>(node)) {
                if (pairs[assign->get_variable().get()].rv &&
                    pairs[assign->get_variable().get()].rv->get_friendly_name() == pair.second.rv->get_friendly_name()) {
                    break;
                }
            }

            for (size_t i = 0; i < node->outputs().size(); i++) {
                for (auto& out_node : node->get_output_target_inputs(i)) {
                    bfs_queue.push_back({out_node.get_node()->shared_from_this() , 0, 0});
                }
            }

            bfs_queue.erase(bfs_queue.begin());
        }

        NodeVector extracted_nodes;
        for (auto it = all_extracted_nodes.rbegin(); it != all_extracted_nodes.rend(); ++it) {
            auto temp_node = it->get();
            bool in_model = false;
            std::string temp_node_name = temp_node->get_friendly_name();
            for (size_t i = 0; i < temp_node->outputs().size(); i++) {
                auto target_inputs = temp_node->output(i).get_target_inputs();
                for (auto& node : extracted_nodes) {
                    std::string analyzed_node = node->get_friendly_name();
                    for (auto& input : node->inputs()) {
                        if (target_inputs.count(input)) {
                            in_model = true;
                            break;
                        }
                    }
                    if (in_model)
                        break;
                }
                if (in_model)
                    break;
            }
            if (extracted_nodes.size() == 0 || in_model) {
                extracted_nodes.push_back(temp_node->shared_from_this());
            }
        }

        try {
            auto extracted_pattern = ov::util::generate_model(extracted_nodes);
            matched_patterns.push_back({ extracted_pattern.first, extracted_pattern.second, extractor_name });
        } catch(std::exception& e) {
            std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " << e.what() << std::endl;
        }
    }

    return matched_patterns;
}
