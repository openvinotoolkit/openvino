// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace functional {

// todo: reuse in summary
inline std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix = "") {
    std::string op_name = node->get_type_info().name;
    std::string opset_version = node->get_type_info().get_version();
    std::string opset_name = "opset";
    auto pos = opset_version.find(opset_name);
    if (pos != std::string::npos) {
        op_name +=  "-" + opset_version.substr(pos + opset_name.size());
    }
    if (!postfix.empty()) {
        op_name += "_" + postfix;
    }
    return op_name;
}

}  // namespace functional
}  // namespace test
}  // namespace ov
