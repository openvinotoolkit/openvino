// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_import/core/null_node.hpp"

#include <string>

#include "ngraph/node.hpp"
#include "openvino/core/deprecated.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
std::shared_ptr<ov::Node> NullNode::clone_with_new_inputs(const ov::OutputVector& /* new_args */) const {
    return std::make_shared<NullNode>();
}
}  // namespace onnx_import
}  // namespace ngraph

bool ngraph::op::is_null(const ngraph::Node* node) {
    return dynamic_cast<const ngraph::onnx_import::NullNode*>(node) != nullptr;
}

bool ngraph::op::is_null(const std::shared_ptr<ngraph::Node>& node) {
    return is_null(node.get());
}

bool ngraph::op::is_null(const Output<ngraph::Node>& output) {
    return is_null(output.get_node());
}
OPENVINO_SUPPRESS_DEPRECATED_END
