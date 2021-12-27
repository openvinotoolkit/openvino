// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx_import/core/node.hpp>
#include <openvino/frontend/onnx/node_context.hpp>

ov::frontend::onnx::NodeContext::NodeContext(const ov::frontend::onnx::Node& context)
    : ov::frontend::NodeContext(context.op_type()),
      m_context(context),
      m_inputs(context.get_ng_inputs()) {}

ov::Output<ov::Node> ov::frontend::onnx::NodeContext::get_input(int port_idx) const {
    return m_inputs.at(port_idx);
}

ov::Any ov::frontend::onnx::NodeContext::get_attribute_as_any(const std::string& name) const {
    return m_context.get_attribute_value<ov::Any>(name);
}

size_t ov::frontend::onnx::NodeContext::get_input_size() const {
    return m_inputs.size();
}
