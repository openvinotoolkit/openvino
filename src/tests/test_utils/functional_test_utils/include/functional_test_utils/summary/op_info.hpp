// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace test {
namespace functional {

// todo: reuse in summary
std::string get_node_version(const std::shared_ptr<ov::Node>& node, const std::string& postfix = "");
std::string get_node_version(const ov::NodeTypeInfo& node_type_info);

}  // namespace functional
}  // namespace test
}  // namespace ov
