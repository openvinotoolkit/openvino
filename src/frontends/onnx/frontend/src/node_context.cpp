// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx_import/core/node.hpp>
#include <openvino/frontend/onnx/node_context.hpp>
#include <utils/common.hpp>

ov::frontend::onnx::NodeContext::NodeContext(const ov::frontend::onnx::Node& context)
    : ov::frontend::NodeContext(context.op_type()),
      m_context(context),
      m_inputs(context.get_ng_inputs()) {}

ov::Output<ov::Node> ov::frontend::onnx::NodeContext::get_input(int port_idx) const {
    return m_inputs.at(port_idx);
}

ov::Any ov::frontend::onnx::NodeContext::get_attribute_as_any(const std::string& name) const {
    try {
        return m_context.get_attribute_value<ov::Any>(name);
    } catch (ngraph::onnx_import::error::node::UnknownAttribute& e) {
        return ov::Any();
    }
}

size_t ov::frontend::onnx::NodeContext::get_input_size() const {
    return m_inputs.size();
}

ov::Any ov::frontend::onnx::NodeContext::apply_additional_conversion_rules(const ov::Any& data,
                                                                           const std::type_info& type_info) const {
    if (data.is<int64_t>() && type_info == typeid(ov::element::Type)) {
        return ngraph::onnx_import::common::get_ngraph_element_type(data.as<int64_t>());
    }

    // no conversion rules found
    return data;
}
