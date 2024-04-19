//*****************************************************************************
// Copyright (C) 2017-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "onnx_framework_node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
std::shared_ptr<ov::Node> ONNXFrameworkNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<ONNXFrameworkNode>(m_node, inputs);
}

std::shared_ptr<ov::Node> ONNXSubgraphFrameworkNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<ONNXSubgraphFrameworkNode>(m_node, m_models, inputs);
}

std::shared_ptr<ov::Node> NotSupportedONNXNode::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    const auto& attrs = get_attrs();
    std::string error_message = attrs.at(failed_conversion_key);
    return std::make_shared<NotSupportedONNXNode>(inputs,
                                                  get_output_size(),
                                                  attrs.get_opset_name(),
                                                  attrs.get_type_name(),
                                                  error_message);
}

bool NotSupportedONNXNode::visit_attributes(ov::AttributeVisitor& visitor) {
    const auto& attrs = get_attrs();
    auto domain = attrs.get_opset_name();
    auto op_type = attrs.get_type_name();
    visitor.on_attribute("ONNX_META_domain", domain);
    visitor.on_attribute("ONNX_META_type", op_type);
    return true;
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
