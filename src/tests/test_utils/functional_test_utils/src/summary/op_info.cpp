// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/op_info.hpp"

namespace ov {
namespace test {
namespace functional {

std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix) {
    const auto& node_type_info = node->get_type_info();
    auto op_name = get_node_version(node_type_info);
    if (!postfix.empty()) {
        op_name += "_" + postfix;
    }
    return op_name;
}

std::string get_node_version(const ov::NodeTypeInfo& node_type_info) {
    std::string op_name = node_type_info.name + std::string("-");
    std::string opset_version = node_type_info.get_version();
    std::string opset_name = "opset";
    auto pos = opset_version.find(opset_name);
    if (pos != std::string::npos) {
        op_name += opset_version.substr(pos + opset_name.size());
    }
    return op_name;
}

}  // namespace functional
}  // namespace test
}  // namespace ov
