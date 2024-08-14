// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/node_util.hpp"

namespace ov {
namespace op {
namespace util {
void set_name(ov::Node& node, const std::string& name, size_t output_port) {
    node.set_friendly_name(name);
    node.get_output_tensor(output_port).set_names({name});
}
}  // namespace util
}  // namespace op
}  // namespace ov
