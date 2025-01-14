// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "matchers/subgraph/read_value_assign.hpp"
#include "utils/model.hpp"

#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"

using namespace ov::tools::subgraph_dumper;

std::vector<ReadValueAssignExtractor::ExtractedPattern>
ReadValueAssignExtractor::extract(const std::shared_ptr<ov::Model> &model) {
    struct ReadValuePairs {
        int cnt_assign = 0;
        int cnt_read_val = 0;
        std::shared_ptr<ov::Node> rv;
        std::string variable_id;
    };
    std::map<ov::op::util::Variable::Ptr, ReadValuePairs>  pairs;
    for (auto& node : model->get_ordered_ops()) {
        if (const auto& assign = ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
            pairs[assign->get_variable()].cnt_assign++;
            pairs[assign->get_variable()].variable_id = assign->get_variable()->get_info().variable_id;
        } else if (const auto& read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
            pairs[read_value->get_variable()].cnt_read_val++;
            pairs[read_value->get_variable()].rv = read_value;
            pairs[read_value->get_variable()].variable_id = read_value->get_variable()->get_info().variable_id;
        }
    }

    std::vector<ReadValueAssignExtractor::ExtractedPattern> matched_patterns;
    for (auto& pair : pairs) {
        if (pair.second.cnt_assign != 1 || pair.second.cnt_read_val != 1) {
            std::cout << "[ WARNING ] Model is incorrect. Assign and ReadValue operations must be in pairs. ";
            std::cout << "Check operations with id: " << pair.second.variable_id << std::endl;
            continue;
        }
        // Breadth first search will find all nodes from ReadValue to Assign from pair
        NodeVector bfs_queue;
        bfs_queue.push_back(pair.second.rv);

        std::vector<std::shared_ptr<ov::Node>> all_extracted_nodes;
        while (bfs_queue.size() != 0) {
            auto node = bfs_queue.front();
            all_extracted_nodes.push_back(node);
            if (const auto& assign = ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
                if (assign->get_variable()->get_info().variable_id == pair.second.variable_id) {
                    break;
                }
            }

            for (size_t i = 0; i < node->outputs().size(); i++) {
                for (auto& out_node : node->get_output_target_inputs(i)) {
                    bfs_queue.push_back(out_node.get_node()->shared_from_this());
                }
            }

            bfs_queue.erase(bfs_queue.begin());
        }

        // Reduce nodes, cross through nodes from Assign to ReadValue and remove all not essential
        NodeVector extracted_nodes;
        for (auto it = all_extracted_nodes.rbegin(); it != all_extracted_nodes.rend(); ++it) {
            auto temp_node = it->get();
            bool in_model = false;
            std::string temp_node_name = temp_node->get_friendly_name();
            for (size_t i = 0; i < temp_node->outputs().size(); i++) {
                auto target_inputs = temp_node->output(i).get_target_inputs();
                for (auto& extr_node : extracted_nodes) {
                    for (auto& input : extr_node->inputs()) {
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
