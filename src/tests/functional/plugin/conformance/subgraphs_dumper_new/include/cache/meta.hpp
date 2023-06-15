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

static unsigned long MIN_MODEL_PRIORITY = std::numeric_limits<unsigned long>::max();
static unsigned long MAX_MODEL_PRIORITY = std::numeric_limits<unsigned long>::min();

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

    bool operator==(const InputInfo& input_info_ref) const {
        return this->is_const == input_info_ref.is_const && this->ranges.max == input_info_ref.ranges.max && this->ranges.min == input_info_ref.ranges.min;
    }
};

struct ModelInfo {
    std::vector<std::string> model_paths;
    size_t this_op_cnt, total_op_cnt, model_priority;

    ModelInfo(const std::string& model_path = "", size_t total_ops_in_model = 1, size_t _model_priority = 1) :
        total_op_cnt(total_ops_in_model), model_paths({model_path}),
        this_op_cnt(1), model_priority(_model_priority) {
        MIN_MODEL_PRIORITY = MAX_MODEL_PRIORITY = total_op_cnt * this_op_cnt * model_priority;
    };

    bool operator==(const ModelInfo& model_info_ref) const {
        if (this->model_priority != model_info_ref.model_priority || this->this_op_cnt != model_info_ref.this_op_cnt ||
            this->total_op_cnt != model_info_ref.total_op_cnt || this->model_paths.size() != model_info_ref.model_paths.size()) {
            return false;
        }
        for (const auto& model_path : this->model_paths) {
            if (std::find(model_info_ref.model_paths.begin(), model_info_ref.model_paths.end(), model_path) == model_info_ref.model_paths.end()) {
                return false;
            }
        }
        return true;
    }
};

class MetaInfo {
public:
    MetaInfo(const std::string& model_path = "", const std::map<std::string, InputInfo>& _input_info = {}, size_t total_op_cnt = 1);
    void serialize(const std::string& serialization_path);
    void update(const std::string& model_path, const std::map<std::string, InputInfo>& _input_info, size_t _total_op_cnt = 1);
    std::map<std::string, InputInfo> get_input_info();
    std::map<std::string, ModelInfo> get_model_info();

protected:
    // { input_node_name: input_info }
    std::map<std::string, InputInfo> input_info;
    // { model_name: model_paths, this_op/graph_cnt, total_op_cnt, model_priority}
    std::map<std::string, ModelInfo> model_info;
    
    double get_graph_priority();
    unsigned long get_abs_graph_priority();
    std::string get_model_name_by_path(const std::string& model_path);
};

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
