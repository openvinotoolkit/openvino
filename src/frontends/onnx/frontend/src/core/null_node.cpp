// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/core/null_node.hpp"

#include <string>

#include "openvino/core/deprecated.hpp"
#include "openvino/core/node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
std::shared_ptr<ov::Node> NullNode::clone_with_new_inputs(const ov::OutputVector& /* new_args */) const {
    return std::make_shared<NullNode>();
}
}  // namespace onnx_import
}  // namespace ngraph

bool ov::op::util::is_null(const ov::Node* node) {
    return dynamic_cast<const ngraph::onnx_import::NullNode*>(node) != nullptr;
}

bool ov::op::util::is_null(const std::shared_ptr<ov::Node>& node) {
    return is_null(node.get());
}

bool ov::op::util::is_null(const Output<ov::Node>& output) {
    return is_null(output.get_node());
}
OPENVINO_SUPPRESS_DEPRECATED_END
