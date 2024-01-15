// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
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

std::unordered_set<std::string>
FusedNamesExtractor::extract_compiled_model_names(const std::shared_ptr<ov::Model>& model) {
    auto compiled_model = ov::util::core->compile_model(model, device);
    std::unordered_set<std::string> compiled_op_name;
    for (const auto& compiled_op : compiled_model.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = compiled_op->get_rt_info();
        if (rt_info.count("originalLayersNames")) {
            compiled_op_name.insert(rt_info.find("originalLayersNames")->second.as<std::string>());
        }
    }
    return compiled_op_name;
}

FusedNamesExtractor::FusedNamesExtractor(const std::string& device) {
    set_target_device(device);
}

std::vector<FusedNamesExtractor::ExtractedPattern>
FusedNamesExtractor::extract(const std::shared_ptr<ov::Model> &model) {
    auto compiled_op_name = extract_compiled_model_names(model);
    std::vector<FusedNamesExtractor::ExtractedPattern> matched_patterns;
    ov::NodeVector nodes;
    for (const auto& op : model->get_ordered_ops()) {
        auto op_name = op->get_friendly_name();
        if (ov::util::is_node_to_skip(op)) {
            continue;
        }
        if (compiled_op_name.count(op_name)) {
            try {
                auto extracted_pattern = ov::util::generate_model(nodes, is_save_const);
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
            if (std::dynamic_pointer_cast<ov::op::v0::TensorIterator>(op)) {
                auto ti = ov::as_type_ptr<ov::op::v0::TensorIterator>(op);
                auto ti_body = ti->get_function();
                auto tmp_res = extract(ti_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v5::Loop>(op)) {
                auto loop = ov::as_type_ptr<ov::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                auto tmp_res = extract(loop_body);
                matched_patterns.insert(matched_patterns.end(), tmp_res.begin(), tmp_res.end());
            } else if (std::dynamic_pointer_cast<ov::op::v8::If>(op)) {
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
}
