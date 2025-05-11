// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <unordered_set>

#include "op_conformance_utils/meta_info/input_info.hpp"
#include "op_conformance_utils/meta_info/model_info.hpp"

namespace ov {
namespace conformance {

class MetaInfo {
public:
    MetaInfo(const std::string& model_path = "",
             const std::map<std::string, InputInfo>& _input_info = {},
             size_t total_op_cnt = 1,
             size_t this_op_cnt = 1,
             const std::string& extractor = "",
             size_t model_priority = 1);
    MetaInfo(std::map<std::string, InputInfo> _in_info,
             std::map<std::string, ModelInfo> _model_info,
             std::unordered_set<std::string> _extractors,
             double _graph_priority = 0) :
             model_info(_model_info),
             input_info(_in_info),
             extractors(_extractors),
             graph_priority(_graph_priority) {};
    void serialize(const std::string& serialization_path);
    void update(const std::string& model_path,
                const std::map<std::string, InputInfo>& _input_info,
                size_t _total_op_cnt = 1,
                size_t _this_op_cnt = 1,
                const std::string& extractor = "",
                const std::vector<std::string>& ignored_inputs = {});
    std::map<std::string, InputInfo> get_input_info() const;
    void set_input_info(const std::map<std::string, InputInfo>& new_in_info) { 
        input_info.clear();
        input_info = new_in_info;
    };
    std::map<std::string, ModelInfo> get_model_info() const;
    std::string get_any_extractor() const { return *extractors.begin(); }
    double get_graph_priority();

    static MetaInfo read_meta_from_file(const std::string& meta_path, bool read_priority = false);

protected:
    // { input_node_name: input_info }
    std::map<std::string, InputInfo> input_info;
    // { model_name: model_paths, this_op/graph_cnt, total_op_cnt, model_priority}
    std::map<std::string, ModelInfo> model_info;
    // { extractors }
    std::unordered_set<std::string> extractors;
    double graph_priority = 0;

    // to store model priority ranges to normilize graph_priority
    static unsigned long MAX_MODEL_PRIORITY;
    static unsigned long MIN_MODEL_PRIORITY;

    std::string get_model_name_by_path(const std::string& model_path);

    // get abs priority graph before normalization
    unsigned long get_abs_graph_priority();
};

}  // namespace conformance
}  // namespace ov
