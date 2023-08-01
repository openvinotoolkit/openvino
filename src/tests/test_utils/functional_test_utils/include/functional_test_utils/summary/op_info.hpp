// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace functional {

// todo: reuse in summary
std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix = "");
}  // namespace functional
}  // namespace test
}  // namespace ov


// todo: remove these structure after remove old subgraphs dumper
namespace LayerTestsUtils {

struct ModelInfo {
    size_t unique_op_cnt;
    // model_path, op_cnt
    std::map<std::string, size_t> model_paths;

    ModelInfo(size_t _op_cnt = 0, const std::map<std::string, size_t>& _model_paths = {{}});
};

struct PortInfo {
    double min;
    double max;
    bool convert_to_const;

    PortInfo(double min, double max, bool convert_to_const);
    PortInfo();
};

struct OPInfo {
    std::map<std::string, ModelInfo> found_in_models;
    std::map<size_t, PortInfo> ports_info;

    OPInfo(const std::string& source_model, const std::string& model_path, size_t total_op_cnt = 0);

    OPInfo() = default;
};
} // namespace LayerTestsUtils
