// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/meta/input_info.hpp"
#include "cache/meta/model_info.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

class MetaInfo {
public:
    MetaInfo(const std::string& model_path = "", const std::map<std::string, InputInfo>& _input_info = {}, size_t total_op_cnt = 1, size_t model_priority = 1);
    void serialize(const std::string& serialization_path);
    void update(const std::string& model_path, const std::map<std::string, InputInfo>& _input_info,
                size_t _total_op_cnt = 1, const std::vector<std::string>& ignored_inputs = {});
    std::map<std::string, InputInfo> get_input_info();
    std::map<std::string, ModelInfo> get_model_info();

protected:
    // { input_node_name: input_info }
    std::map<std::string, InputInfo> input_info;
    // { model_name: model_paths, this_op/graph_cnt, total_op_cnt, model_priority}
    std::map<std::string, ModelInfo> model_info;

    // to store model priority ranges to normilize graph_priority
    static unsigned long MAX_MODEL_PRIORITY;
    static unsigned long MIN_MODEL_PRIORITY;

    double get_graph_priority();
    std::string get_model_name_by_path(const std::string& model_path);

    // get abs priority graph before normalization
    unsigned long get_abs_graph_priority();
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
