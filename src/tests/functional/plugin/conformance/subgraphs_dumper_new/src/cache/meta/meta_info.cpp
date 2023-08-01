// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pugixml.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/meta/meta_info.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

unsigned long MetaInfo::MIN_MODEL_PRIORITY = std::numeric_limits<unsigned long>::max();
unsigned long MetaInfo::MAX_MODEL_PRIORITY = std::numeric_limits<unsigned long>::min();

MetaInfo::MetaInfo(const std::string& _model_path, const std::map<std::string, InputInfo>& _input_info, size_t _total_op_cnt, size_t model_priority) {
    unsigned long tmp_graph_priority = _total_op_cnt * model_priority;
    if (tmp_graph_priority < MIN_MODEL_PRIORITY) MIN_MODEL_PRIORITY = tmp_graph_priority;
    if (tmp_graph_priority > MAX_MODEL_PRIORITY) MAX_MODEL_PRIORITY = tmp_graph_priority;
    if (_model_path != "") {
        model_info.insert({ get_model_name_by_path(_model_path), ModelInfo(_model_path, _total_op_cnt, model_priority) });
    }
    if (!_input_info.empty()) {
        input_info = _input_info;
    }
}

unsigned long MetaInfo::get_abs_graph_priority() {
    unsigned long res = 0;
    for (const auto& model : model_info) {
        res += model.second.total_op_cnt * model.second.this_op_cnt * model.second.model_priority;
    }
    return res;
}

double MetaInfo::get_graph_priority() {
    auto delta = MAX_MODEL_PRIORITY - MIN_MODEL_PRIORITY == 0 ? 1 : MAX_MODEL_PRIORITY - MIN_MODEL_PRIORITY;
    // return normilized graph priority from [0, 1]
    double diff = get_abs_graph_priority() - MIN_MODEL_PRIORITY;
    return diff / delta;
}

void MetaInfo::serialize(const std::string& serialization_path) {
        pugi::xml_document doc;
        pugi::xml_node root = doc.append_child("meta_info");
        pugi::xml_node models = root.append_child("models");
        // todo: iefode: update to prioritize_latest opset
        for (const auto& model : model_info) {
            pugi::xml_node model_node = models.append_child("model");
            model_node.append_attribute("name").set_value(model.first.c_str());
            model_node.append_attribute("this_op_count").set_value(static_cast<unsigned long long>(model.second.this_op_cnt));
            model_node.append_attribute("total_op_count").set_value(static_cast<unsigned long long>(model.second.total_op_cnt));
            for (const auto& model_path : model.second.model_paths) {
                model_node.append_child("path").append_child("model").append_attribute("path").set_value(model_path.c_str());
            }
        }
        double graph_priority = get_graph_priority();
        root.append_child("graph_priority").append_attribute("value").set_value(graph_priority);
        auto ports_info = root.append_child("input_info");
        for (const auto& input : input_info) {
            auto input_node = ports_info.append_child("input");
            input_node.append_attribute("id").set_value(input.first.c_str());
            if (input.second.ranges.min == DEFAULT_MIN_VALUE) {
                input_node.append_attribute("min").set_value("undefined");
            } else {
                input_node.append_attribute("min").set_value(input.second.ranges.min);
            }
            if (input.second.ranges.max == DEFAULT_MAX_VALUE) {
                input_node.append_attribute("max").set_value("undefined");
            } else {
                input_node.append_attribute("max").set_value(input.second.ranges.max);
            }
            input_node.append_attribute("convert_to_const").set_value(input.second.is_const);
        }
        doc.save_file(serialization_path.c_str());
}

void MetaInfo::update(const std::string& _model_path,
                      const std::map<std::string, InputInfo>& _input_info,
                      size_t _total_op_cnt,
                      const std::vector<std::string>& ignored_inputs) {
    if (input_info.size() != _input_info.size()) {
        throw std::runtime_error("Incompatible input info!");
    }
    std::string model_name = get_model_name_by_path(_model_path);
    if (model_info.find(model_name) != model_info.end()) {
        if (model_info.at(model_name).model_paths.find(_model_path) == model_info.at(model_name).model_paths.end()) {
            model_info.at(model_name).model_paths.insert(_model_path);
            model_info.at(model_name).total_op_cnt += _total_op_cnt;
        }
        model_info.at(model_name).this_op_cnt++;
    } else {
        model_info.insert({ model_name, ModelInfo(_model_path, _total_op_cnt) });\
    }
    for (const auto& in : _input_info) {
        if (std::find(ignored_inputs.begin(), ignored_inputs.end(), in.first) != ignored_inputs.begin()) {
            continue;
        }
        if (input_info.find(in.first) == input_info.end()) {
            throw std::runtime_error("Incorrect Input Info!");
        } else if (input_info[in.first].is_const != in.second.is_const) {
            throw std::runtime_error("Try to cast parameter ro constant!");
        } else {
            input_info[in.first] = in.second;
        }
    }
    // update max and mib abs priority to normilize priorities when serialize
    {
        auto abs_graph_priority = get_abs_graph_priority();
        if (abs_graph_priority > MAX_MODEL_PRIORITY) MAX_MODEL_PRIORITY = abs_graph_priority;
        if (abs_graph_priority < MIN_MODEL_PRIORITY) MIN_MODEL_PRIORITY = abs_graph_priority;
    }
}

std::map<std::string, InputInfo> MetaInfo::get_input_info() {
    return input_info;
}

std::map<std::string, ModelInfo> MetaInfo::get_model_info() {
    return model_info;
}

std::string MetaInfo::get_model_name_by_path(const std::string& model_path) {
    auto pos = model_path.rfind(ov::test::utils::FileSeparator);
    auto model_name = ov::test::utils::replaceExt(model_path.substr(pos + 1), "");
    return model_name;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
