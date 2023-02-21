// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace LayerTestsUtils {

struct ModelInfo {
    size_t op_cnt;
    std::set<std::string> model_paths;

    ModelInfo(size_t _op_cnt = 0, const std::set<std::string>& _model_paths = {}) : op_cnt(_op_cnt), model_paths(_model_paths) {}
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

    OPInfo(const std::string& source_model, const std::string& model_path) {
        found_in_models = {{source_model, ModelInfo(1, {model_path})}};
        ports_info = {};
    }

    OPInfo() = default;
};
} // namespace LayerTestsUtils
