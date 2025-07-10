// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/node_util.hpp"

#include "openvino/core/descriptor_tensor.hpp"

namespace ov::op::util {

void set_name(ov::Node& node, const std::string& name, size_t output_port) {
    node.set_friendly_name(name);
    node.get_output_tensor(output_port).set_names({name});
}
}  // namespace ov::op::util

namespace ov::util {

std::string make_default_tensor_name(const Output<const Node>& output) {
    auto default_name = output.get_node()->get_friendly_name();
    if (output.get_node()->get_output_size() > 1) {
        default_name += descriptor::port_separator + std::to_string(output.get_index());
    }
    return default_name;
}
}  // namespace ov::util
