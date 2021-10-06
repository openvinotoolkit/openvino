// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace LayerTestsUtils {
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
    std::string source_model;
    std::map<std::string, size_t> found_in_models;
    std::map<size_t, PortInfo> ports_info;

    OPInfo(const std::string &source_model) : source_model(source_model) {
        found_in_models = {{source_model, 1}};
        ports_info = {};
    }

    OPInfo() = default;
};
} // namespace LayerTestsUtils