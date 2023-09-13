// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/serialization_node.hpp"


namespace ov {
namespace snippets {
namespace op {

SerializationNode::SerializationNode(const ov::OutputVector& args, const std::shared_ptr<lowered::Expression>& expr)
    : Op(args), m_expr(expr) {
    if (!m_expr || !m_expr->get_node())
        OPENVINO_THROW("SerializationNode requires a valid expression with non-null node pointer");
    const auto &node = expr->get_node();
    std::string type = node->get_type_name();
    std::string name = node->get_friendly_name();
    // If node is a parameter, show another type name, so the node will be displayed correctly
    get_rt_info()["layerType"] = type == "Parameter" ? "ParameterLowered" : type;
    set_friendly_name(name);
    constructor_validate_and_infer_types();
}

void SerializationNode::validate_and_infer_types() {
    set_output_type(0, element::f32, {});
}

std::shared_ptr<Node> SerializationNode::clone_with_new_inputs(const OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SerializationNode>(new_args, m_expr);
}

bool SerializationNode::visit_attributes(AttributeVisitor &visitor) {
    std::vector<std::pair<std::string, std::vector<size_t>>> shapes;
    for (size_t i = 0; i < m_expr->get_input_count(); i++) {
        const auto &shape = m_expr->get_input_port_descriptor(i)->get_shape();
        if (!shape.empty())
            shapes.emplace_back("in_shape_" + std::to_string(i), shape);
    }
    for (size_t i = 0; i < m_expr->get_output_count(); i++) {
        const auto &shape = m_expr->get_output_port_descriptor(i)->get_shape();
        if (!shape.empty())
            shapes.emplace_back("out_shape_" + std::to_string(i), shape);
    }

    auto loop_ids = m_expr->get_loop_ids();
    auto rinfo = m_expr->get_reg_info();
    if (!rinfo.first.empty())
        visitor.on_attribute("in_regs", rinfo.first);
    if (!rinfo.second.empty())
        visitor.on_attribute("out_regs", rinfo.second);
    for (auto& s : shapes)
        visitor.on_attribute(s.first, s.second);

    visitor.on_attribute("loop_ids", loop_ids);
    m_expr->get_node()->visit_attributes(visitor);
    return true;
}

} // namespace op
} // namespace snippets
} // namespace ov
