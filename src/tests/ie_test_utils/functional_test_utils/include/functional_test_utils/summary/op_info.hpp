// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace LayerTestsUtils {

struct ModelInfo {
    size_t unique_op_cnt;
    // model_path, op_cnt
    std::map<std::string, size_t> model_paths;


    ModelInfo(size_t _op_cnt = 0, const std::map<std::string, size_t>& _model_paths = {{}})
        : unique_op_cnt(_op_cnt),
          model_paths(_model_paths) {}
};

struct PortInfo {
    double min;
    double max;
    bool convert_to_const;

    PortInfo(double min, double max, bool convert_to_const) : min(min), max(max),
                                                              convert_to_const(convert_to_const) {}
    PortInfo() {
        min = std::numeric_limits<double>::min();
        max = std::numeric_limits<double>::max();
        convert_to_const = false;
    }
};

struct OPInfo {
    std::map<std::string, ModelInfo> found_in_models;
    std::map<size_t, PortInfo> ports_info;

    OPInfo(const std::string& source_model, const std::string& model_path, size_t total_op_cnt = 0) {
        found_in_models = {{source_model, ModelInfo(1, {{model_path, total_op_cnt}})}};
        ports_info = {};
    }

    OPInfo() = default;
};
} // namespace LayerTestsUtils
