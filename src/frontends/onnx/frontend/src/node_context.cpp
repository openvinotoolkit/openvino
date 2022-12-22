// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <onnx_import/core/node.hpp>
#include <openvino/frontend/onnx/node_context.hpp>
#include <utils/common.hpp>

ov::frontend::onnx::NodeContext::NodeContext(const ngraph::onnx_import::Node& context)
    : ov::frontend::NodeContext(context.op_type()),
      m_context(context),
      m_inputs(context.get_ng_inputs()) {}

ov::Output<ov::Node> ov::frontend::onnx::NodeContext::get_input(int port_idx) const {
    return m_inputs.at(port_idx);
}

ov::Any ov::frontend::onnx::NodeContext::get_attribute_as_any(const std::string& name) const {
    try {
        return m_context.get_attribute_value<ov::Any>(name);
    } catch (ngraph::onnx_import::error::node::UnknownAttribute&) {
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
    } else if (data.is<std::vector<int64_t>>() && type_info == typeid(std::vector<ov::element::Type>)) {
        const auto& casted = data.as<std::vector<int64_t>>();
        std::vector<ov::element::Type> types(casted.size());
        for (size_t i = 0; i < casted.size(); ++i) {
            types[i] = ngraph::onnx_import::common::get_ngraph_element_type(casted[i]);
        }
        return types;
    }
    // no conversion rules found
    return data;
}
