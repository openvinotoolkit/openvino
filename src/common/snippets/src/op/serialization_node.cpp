// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/serialization_node.hpp"


namespace ov {
namespace snippets {
namespace op {

SerializationNode::SerializationNode(const ov::OutputVector& args,
                                     const std::shared_ptr<lowered::Expression>& expr,
                                     SerializationMode mode)
    : Op(args),
      m_expr(expr),
      m_mode(mode) {
    OPENVINO_ASSERT(m_expr && m_expr->get_node(), "SerializationNode requires a valid expression with non-null node pointer");
    const auto& node = expr->get_node();
    set_friendly_name(node->get_friendly_name());
    std::string type = node->get_type_name();
    get_rt_info()["layerType"] = type == "Parameter" ? "ParameterLowered" : type;
    constructor_validate_and_infer_types();
}

void SerializationNode::validate_and_infer_types() {
    // If SerializationNode is used for control flow serialization, it always has one output
    // (since it represents a linear execution order)
    if (m_mode == SerializationMode::CONTROL_FLOW) {
        set_output_type(0, element::f32, {});
    } else if (m_mode == SerializationMode::DATA_FLOW) {
        for (size_t i = 0; i < m_expr->get_output_count(); ++i)
            set_output_type(i, element::f32, {});
    }
}

std::shared_ptr<Node> SerializationNode::clone_with_new_inputs(const OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SerializationNode>(new_args, m_expr, m_mode);
}

bool SerializationNode::visit_attributes(AttributeVisitor &visitor) {
    auto is_planar_layout = [](const std::vector<size_t>& layout) {
        for (size_t i = 0; i < layout.size(); ++i)
            if (layout[i] != i) return false;
        return true;
    };
    auto subtensor2str = [](const VectorDims& subtensor) {
        std::stringstream ss;
        for (size_t i = 0; i < subtensor.size(); ++i) {
            const auto& v = subtensor[i];
            const auto v_str = v == lowered::PortDescriptor::ServiceDimensions::FULL_DIM ? "FULL_DIM" :
                               (utils::is_dynamic_value(v) ? "?" : std::to_string(v));
            const auto del = i < subtensor.size() - 1 ? ", " : "";
            ss << v_str << del;
        }
        return ss.str();
    };

    std::vector<size_t> in_regs, out_regs;
    std::vector<std::string> in_reg_types, out_reg_types;
    std::vector<std::pair<std::string, ov::PartialShape>> shapes;
    std::vector<std::pair<std::string, std::string>> subtensors;
    std::vector<std::pair<std::string, std::vector<size_t>>> layouts;
    for (size_t i = 0; i < m_expr->get_input_count(); i++) {
        const auto& desc = m_expr->get_input_port_descriptor(i);
        const auto& shape = desc->get_shape();
        if (!shape.empty())
            shapes.emplace_back("in_shape_" + std::to_string(i), ov::PartialShape(shape));

        const auto& subtensor = desc->get_subtensor();
        if (!subtensor.empty())
            subtensors.emplace_back("in_subtensor_" + std::to_string(i), subtensor2str(subtensor));

        const auto& layout = desc->get_layout();
        if (!layout.empty() && !is_planar_layout(layout))
            layouts.emplace_back("in_layout_" + std::to_string(i), layout);

        in_reg_types.emplace_back(regTypeToStr(desc->get_reg().type));
        in_regs.emplace_back(desc->get_reg().idx);
    }
    for (size_t i = 0; i < m_expr->get_output_count(); i++) {
        const auto& desc = m_expr->get_output_port_descriptor(i);
        const auto& shape = desc->get_shape();
        if (!shape.empty())
            shapes.emplace_back("out_shape_" + std::to_string(i), ov::PartialShape(shape));

        const auto& subtensor = desc->get_subtensor();
        if (!subtensor.empty())
            subtensors.emplace_back("out_subtensor_" + std::to_string(i), subtensor2str(subtensor));

        const auto& layout = desc->get_layout();
        if (!layout.empty() && !is_planar_layout(layout))
            layouts.emplace_back("out_layout_" + std::to_string(i), layout);

        out_reg_types.emplace_back(regTypeToStr(desc->get_reg().type));
        out_regs.emplace_back(desc->get_reg().idx);
    }

    if (!in_regs.empty()) {
        visitor.on_attribute("in_regs", in_regs);
        visitor.on_attribute("in_reg_types", in_reg_types);
    }
    if (!out_regs.empty()) {
        visitor.on_attribute("out_regs", out_regs);
        visitor.on_attribute("out_reg_types", out_reg_types);
    }
    for (auto& s : shapes)
        visitor.on_attribute(s.first, s.second);
    for (auto& s : subtensors)
        visitor.on_attribute(s.first, s.second);
    for (auto& s : layouts)
        visitor.on_attribute(s.first, s.second);

    auto loop_ids = m_expr->get_loop_ids();
    visitor.on_attribute("loop_ids", loop_ids);
    m_expr->get_node()->visit_attributes(visitor);
    return true;
}

} // namespace op
} // namespace snippets
} // namespace ov
