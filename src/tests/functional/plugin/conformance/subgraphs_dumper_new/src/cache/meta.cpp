// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pugixml.hpp"

#include "common_test_utils/file_utils.hpp"

#include "cache/meta.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

MetaInfo::MetaInfo(const std::string& _model_path, const std::map<std::string, InputInfo>& _input_info) {
    std::string model_name = get_model_name_by_path(_model_path);
    model_path = {{ model_name, { _model_path } }};
    occurence_cnt = {{ model_name, 1 }};
    input_info = _input_info;
}

double MetaInfo::get_graph_priority() {}

void MetaInfo::serialize(const std::string& serialization_path) {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("meta_info");
    pugi::xml_node models = root.append_child("models");
    // todo: iefode: update to prioritize_latest opset
    for (const auto& model_name : model_path) {
        pugi::xml_node model_node = models.append_child("model");
        model_node.append_attribute("name").set_value(model_name.first.c_str());
        model_node.append_attribute("count").set_value(occurence_cnt[model_name.first]);
        for (const auto& model : model_name.second) {
            model_node.append_child("path").append_attribute("model").set_value(model.c_str());
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

void MetaInfo::update(const std::string& _model_path, const std::map<std::string, InputInfo>& _input_info) {
    std::string model_name = get_model_name_by_path(_model_path);
    if (model_path.find(model_name) != model_path.end()) {
        model_path[model_name].push_back(_model_path);
        occurence_cnt[model_name]++;
    } else {
        model_path.insert({ model_name, { _model_path } });
        occurence_cnt.insert({ model_name, 1 });
    }
    for (const auto& in : _input_info) {
        if (input_info.find(in.first) == input_info.end()) {
            throw std::runtime_error("Incorrect Input Info!");
        } else {
            input_info[in.first] = in.second;
        }
    }
}

std::map<std::string, InputInfo> MetaInfo::get_input_info() {
    return input_info;
}

std::string MetaInfo::get_model_name_by_path(const std::string& model_path) {
    auto pos = model_path.rfind(CommonTestUtils::FileSeparator);
    auto model_name = pos == std::string::npos ? model_path : CommonTestUtils::replaceExt(model_path.substr(pos + 1), "");
    return model_name;
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
