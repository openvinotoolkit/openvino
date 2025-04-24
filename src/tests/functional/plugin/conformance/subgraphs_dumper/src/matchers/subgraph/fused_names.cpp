// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/util/file_util.hpp"

#include "matchers/subgraph/fused_names.hpp"
#include "utils/model.hpp"
#include "utils/cache.hpp"

using namespace ov::tools::subgraph_dumper;

void FusedNamesExtractor::set_target_device(const std::string& _device) {
    auto available_devices = ov::util::core->get_available_devices();
    if (_device == std::string("TEMPLATE") &&
        std::find(available_devices.begin(), available_devices.end(), _device) == available_devices.end()) {
        auto plugin_path = ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), std::string("openvino_template_plugin") + OV_BUILD_POSTFIX);
        if (!ov::util::file_exists(plugin_path)) {
            throw std::runtime_error("[ WARNING ][ GRAPH CACHE ] Plugin: " + plugin_path + " does not exists!");
        }
        ov::util::core->register_plugin(plugin_path, _device);
        available_devices = ov::util::core->get_available_devices();
    }
    if (_device.empty() && !available_devices.empty()) {
        device = available_devices.front();
        std::cout << "[ WARNING ][ GRAPH CACHE ] " << device <<
            " will be used for `fused_names` extractor" << std::endl;
        return;
    } else if (std::find(available_devices.begin(),
               available_devices.end(),
               _device) == available_devices.end()) {
        std::string message = "Incorrect device ";
        message += _device;
        message += " to enable `fused_names` extractor! Available devices: {";
        for (const auto& device : available_devices) {
            message += device;
            message += ",";
        }
        message += "}";
        throw std::runtime_error(message);
    }
    device = _device;
    std::cout << "[ INFO ][ GRAPH CACHE ] " << device << " is using for `fused_names` extractor" << std::endl;
}

inline std::string get_original_layer_name(const std::shared_ptr<ov::Node>& node) {
    std::string original_layer_name = node->get_friendly_name();
    const auto& rt_info = node->get_rt_info();
    if (rt_info.count("originalLayersNames")) {
        original_layer_name = rt_info.find("originalLayersNames")->second.as<std::string>();
    }
    return original_layer_name;
}

// shape of
inline bool is_node_transformed(const std::shared_ptr<ov::Node>& node) {
    const auto compiled_node_name = node->get_friendly_name();
    if (get_original_layer_name(node) != compiled_node_name) {
        return true;
    }
    for (const auto& in_value : node->input_values()) {
        const auto in_node = in_value.get_node_shared_ptr();
        if (in_node->get_friendly_name() != get_original_layer_name(in_node)) {
            return true;
        }
    }
    for (const auto& output : node->outputs()) {
        for (const auto& target_input : output.get_target_inputs()) {
            auto target_in_node = target_input.get_node()->shared_from_this();
            if (target_in_node->get_friendly_name() != get_original_layer_name(target_in_node)) {
                return true;
            }
        }
    }
    return false;
}

std::unordered_set<std::string>
FusedNamesExtractor::extract_not_trasformed_node_names(const std::shared_ptr<ov::Model>& model) {
    auto compiled_model = ov::util::core->compile_model(model, device);
    std::unordered_set<std::string> not_transformed_nodes;
    for (const auto& compiled_op : compiled_model.get_runtime_model()->get_ordered_ops()) {
        if (!is_node_transformed(compiled_op)) {
            not_transformed_nodes.insert(compiled_op->get_friendly_name());
        }
    }
    for (const auto& op : model->get_ordered_ops()) {
        const auto op_name = op->get_friendly_name();
        if (!not_transformed_nodes.count(op_name)) {
            continue;
        }
        if (op->get_type_info().is_castable(ov::op::v0::ShapeOf::get_type_info_static()) ||
            op->get_type_info().is_castable(ov::op::v3::ShapeOf::get_type_info_static())) {
            not_transformed_nodes.erase(op_name);
        }
    }
    return not_transformed_nodes;
}

FusedNamesExtractor::FusedNamesExtractor(const std::string& device) {
    set_target_device(device);
}

// std::vector<FusedNamesExtractor::NodeDescriptor>
// FusedNamesExtractor::extract_transformed_nodes(const std::shared_ptr<ov::Model>& model) {
//     std::vector<NodeDescriptor> transformed_ops;
//     const auto not_transformed_ops = extract_not_trasformed_node_names(model);
//     {
//         auto ordered_ops = model->get_ordered_ops();
//         std::reverse(ordered_ops.begin(), ordered_ops.end());
//         for (size_t i = 0; i < ordered_ops.size(); ++i) {
//             const auto op = ordered_ops[i];
//             if (not_transformed_ops.count(op->get_friendly_name()) || util::is_node_to_skip(op)) {
//                 continue;
//             }
//             transformed_ops.push_back(op);
//         }
//     }
//     for (size_t i = 0; i < transformed_ops.size(); ++i) {
//         auto& transformed_op = transformed_ops[i];
//         for (const auto& input_value : transformed_op.node->input_values()) {
//             const auto input_node = input_value.get_node_shared_ptr();
//             if (not_transformed_ops.count(input_node->get_friendly_name()) || util::is_node_to_skip(input_node)) {
//                 continue;
//             }
//             for (size_t j = i; j < transformed_ops.size(); ++j) {
//                 if (transformed_ops[j].node == input_node) {
//                     transformed_op.input_idx.insert(j);
//                     break;
//                 }
//             }
//             for (const auto& j : transformed_op.input_idx) {
//                 transformed_ops[j].output_idx.insert(i);
//             }
//         }
//     }
//     return transformed_ops;
// }

// std::unordered_map<size_t, ov::NodeVector>
// FusedNamesExtractor::label_subgraphs(std::vector<FusedNamesExtractor::NodeDescriptor>& transformed_ops) {
//     std::unordered_map<size_t, ov::NodeVector> subgraphs;
//     const auto backward_propagation = [&transformed_ops, &subgraphs](size_t node_id) {
//         std::deque<size_t> deque{node_id};
//         const size_t subgraph_id = transformed_ops[node_id].subgraph_id;
//         while (!deque.empty()) {
//             size_t front_node_id = deque.front();
//             deque.pop_front();
//             for (const auto& out_node_id : transformed_ops[front_node_id].output_idx) {
//                 if (transformed_ops[out_node_id].subgraph_id == subgraph_id) {
//                     continue;
//                 }
//                 if (subgraphs.count(transformed_ops[out_node_id].subgraph_id)) {
//                     const auto node_vector = subgraphs[transformed_ops[out_node_id].subgraph_id];
//                     subgraphs[subgraph_id].insert(subgraphs[subgraph_id].end(), node_vector.begin(), node_vector.end());
//                     subgraphs.erase(transformed_ops[out_node_id].subgraph_id);
//                 }
//                 transformed_ops[out_node_id].subgraph_id = subgraph_id;
//                 deque.push_back(out_node_id);
//             }
//         }
//     };

//     size_t subgraph_id = 0;
//     for (size_t i = 0; i < transformed_ops.size(); ++i) {
//         auto& transformed_op = transformed_ops[i];
//         if (!transformed_op.is_defined()) {
//             transformed_op.subgraph_id = subgraph_id++;
//             subgraphs.insert({transformed_op.subgraph_id, {transformed_op.node}});
//         } else if (transformed_op.output_idx.size() > 1) {
//             backward_propagation(i);
//         }
//         for (const auto& in_idx : transformed_op.input_idx) {
//             transformed_ops[in_idx].subgraph_id = transformed_op.subgraph_id;
//             subgraphs[transformed_op.subgraph_id].push_back(transformed_ops[in_idx].node);
//         }
//     }
//     return subgraphs;
// }

std::vector<FusedNamesExtractor::ExtractedPattern>
FusedNamesExtractor::extract(const std::shared_ptr<ov::Model> &model) {
    std::vector<FusedNamesExtractor::ExtractedPattern> matched_patterns;
    const auto not_transformed_nodes = extract_not_trasformed_node_names(model);
    ov::NodeVector nodes;
    for (const auto& op : model->get_ordered_ops()) {
        auto op_name = op->get_friendly_name();
        if (ov::util::is_node_to_skip(op)) {
            continue;
        }
        if (not_transformed_nodes.count(op_name)) {
            try {
                auto extracted_pattern = ov::util::generate_model(nodes, true);
                matched_patterns.push_back({ extracted_pattern.first, extracted_pattern.second, extractor_name });
            } catch(std::exception& e) {
                if (std::string(e.what()).find("Incorrect node number to create model") == std::string::npos) {
                    // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
                }
            }
            nodes.clear();
        } else {
            nodes.push_back(op);
        }
        if (is_extract_body) {
            if (ov::as_type_ptr<ov::op::v0::TensorIterator>(op)) {
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                auto tmp_res = extract(ti_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (ov::as_type_ptr<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                auto tmp_res = extract(loop_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (ov::as_type_ptr<ov::op::v8::If>(op)) {
                auto if_op = ov::as_type_ptr<ov::op::v8::If>(op);
                std::vector<std::shared_ptr<ov::Model>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    auto tmp_res = extract(if_body);
                    matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
                }
            }
        }
    }
    try {
        auto extracted_pattern = ov::util::generate_model(nodes, is_save_const);
        matched_patterns.push_back({ extracted_pattern.first, extracted_pattern.second, extractor_name });
    } catch(std::exception& e) {
        if (std::string(e.what()).find("Incorrect node number to create model") == std::string::npos) {
            // std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
        }
    }

    return matched_patterns;

    // possible solution to extract transformed graphs
    // problems: takes a lot of time + process ops with bodies
    // auto transformed_ops = extract_transformed_nodes(model);
    // for (auto& subgraph : label_subgraphs(transformed_ops)) {
    //     try {
    //         auto extracted_pattern = ov::util::generate_model(subgraph.second, is_save_const);
    //         matched_patterns.push_back({ extracted_pattern.first, extracted_pattern.second, extractor_name });
    //     } catch(std::exception& e) {
    //         if (std::string(e.what()).find("Incorrect node number to create model") == std::string::npos) {
    //             std::cout << "[ WARNING ] Impossible to generate network and add to GraphCache: " <<e.what() << std::endl;
    //         }
    //     }
    // }
    // return matched_patterns;
}
