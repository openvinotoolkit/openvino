// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onnx_framework_node.hpp"

namespace ov::frontend::onnx {
std::shared_ptr<ov::Node> ONNXFrameworkNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<ONNXFrameworkNode>(m_node, inputs);
}

std::shared_ptr<ov::Node> ONNXSubgraphFrameworkNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<ONNXSubgraphFrameworkNode>(m_node, m_models, inputs);
}

std::shared_ptr<ov::Node> NotSupportedONNXNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    const auto& attrs = get_attrs();
    return std::make_shared<NotSupportedONNXNode>(inputs,
                                                  get_output_size(),
                                                  attrs.get_opset_name(),
                                                  attrs.get_type_name(),
                                                  opset_version(),
                                                  additional_error_message());
}

bool NotSupportedONNXNode::visit_attributes(ov::AttributeVisitor& visitor) {
    const auto& attrs = get_attrs();
    auto domain = attrs.get_opset_name();
    auto op_type = attrs.get_type_name();
    visitor.on_attribute("ONNX_META_domain", domain);
    visitor.on_attribute("ONNX_META_type", op_type);
    return true;
}

}  // namespace ov::frontend::onnx
