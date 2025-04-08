// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"

#include <string>

#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
std::shared_ptr<ov::Node> NullNode::clone_with_new_inputs(const ov::OutputVector& /* new_args */) const {
    return std::make_shared<NullNode>();
}
}  // namespace onnx
}  // namespace frontend
}  // namespace ov

bool ov::op::util::is_null(const ov::Node* node) {
    return ov::as_type<const ov::frontend::onnx::NullNode>(node) != nullptr;
}

bool ov::op::util::is_null(const std::shared_ptr<ov::Node>& node) {
    return is_null(node.get());
}

bool ov::op::util::is_null(const Output<ov::Node>& output) {
    return is_null(output.get_node());
}
