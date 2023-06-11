// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

namespace ov {
namespace tools {
namespace subgraph_dumper {

constexpr double DEFAULT_MIN_VALUE = std::numeric_limits<double>::min();
constexpr double DEFAULT_MAX_VALUE = std::numeric_limits<double>::max();

struct InputInfo {
    struct Range {
        double min, max;

        Range(double in_min, double in_max) : min(in_min), max(in_max) {}

        Range& operator=(const Range& ranges) {
            if (ranges.max != DEFAULT_MAX_VALUE) {
                this->max = this->max != DEFAULT_MAX_VALUE ? std::max(this->max, ranges.max) : ranges.max;
            }
            if (ranges.min != DEFAULT_MIN_VALUE) {
                this->min = this->min != DEFAULT_MIN_VALUE ? std::min(this->min, ranges.min) : ranges.min;
            }
            return *this;
        }
    };

    Range ranges;
    bool is_const;

    InputInfo(double in_min = DEFAULT_MIN_VALUE,
              double in_max = DEFAULT_MAX_VALUE,
              bool in_is_const = false) :
              is_const(in_is_const),
              ranges(Range(in_min, in_max)) {}
};

class MetaInfo {
public:
    MetaInfo(const std::string& model_path = "", const std::map<std::string, InputInfo>& _input_info = {});
    void serialize(const std::string& serialization_path);
    void update(const std::string& model_path, const std::map<std::string, InputInfo>& _input_info);
    std::map<std::string, InputInfo> get_input_info();
    std::map<std::string, std::vector<std::string>> get_model_path();
    std::map<std::string, size_t> get_occurence_cnt();

protected:
    // { input_node_name: input_info }
    std::map<std::string, InputInfo> input_info;
    // { model_name: [ model_paths ] }
    std::map<std::string, std::vector<std::string>> model_path;
    // { model_name: op/graph_occurence_in_model }
    std::map<std::string, size_t> occurence_cnt;

    std::string get_model_name_by_path(const std::string& model_path);
    double get_graph_priority();
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
