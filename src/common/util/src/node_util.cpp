// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/node_util.hpp"

namespace ov {
namespace util {
std::shared_ptr<ov::Node> set_name(std::shared_ptr<ov::Node> node, const std::string& name, size_t output_idx) {
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() >= output_idx);
    node->get_output_tensor(output_idx).set_names({name});
    return node;
}
}  // namespace util
}  // namespace ov
