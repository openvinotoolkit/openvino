// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/core/null_node.hpp"

#include <string>

#include "openvino/core/node.hpp"

namespace ov {
namespace onnx_import {
constexpr NodeTypeInfo NullNode::type_info;

std::shared_ptr<ov::Node> NullNode::clone_with_new_inputs(const OutputVector& /* new_args */) const {
    return std::make_shared<NullNode>();
}
}  // namespace onnx_import
}  // namespace ov

bool ov::op::is_null(const ov::Node* node) {
    return dynamic_cast<const ov::onnx_import::NullNode*>(node) != nullptr;
}

bool ov::op::is_null(const std::shared_ptr<ov::Node>& node) {
    return is_null(node.get());
}

bool ov::op::is_null(const Output<ov::Node>& output) {
    return is_null(output.get_node());
}
