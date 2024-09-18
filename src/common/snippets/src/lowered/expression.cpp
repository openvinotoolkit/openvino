// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/expression.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {

Expression::Expression(const std::shared_ptr<Node>& n, const std::shared_ptr<IShapeInferSnippetsFactory>& factory, bool need_shape_infer)
        : m_source_node{n}, m_emitter{nullptr}, m_input_port_connectors{}, m_output_port_connectors{},
          m_shapeInference(make_shape_inference(n, factory)), m_need_shape_infer(need_shape_infer) {
    m_input_port_descriptors.reserve(n->get_input_size());
    m_output_port_descriptors.reserve(n->get_output_size());
    for (const auto& input : n->inputs()) {
        m_input_port_descriptors.push_back(PortDescriptorUtils::get_port_descriptor_ptr(input));
    }
    for (const auto& output : n->outputs()) {
        m_output_port_descriptors.push_back(PortDescriptorUtils::get_port_descriptor_ptr(output));
    }
}

const PortConnectorPtr& Expression::get_input_port_connector(size_t i) const {
    OPENVINO_ASSERT(i < m_input_port_connectors.size(), "Failed to get input port connector: target input port must be less than input count!");
    return m_input_port_connectors[i];
}
const PortConnectorPtr& Expression::get_output_port_connector(size_t i) const {
    OPENVINO_ASSERT(i < m_output_port_connectors.size(), "Failed to get output port connector: target output port must be less than output count!");
    return m_output_port_connectors[i];
}

const PortDescriptorPtr& Expression::get_input_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_input_port_descriptors.size(), "Failed to get input port descriptor: target input port must be less than input count!");
    return m_input_port_descriptors[i];
}
const PortDescriptorPtr& Expression::get_output_port_descriptor(size_t i) const {
    OPENVINO_ASSERT(i < m_output_port_descriptors.size(), "Failed to get output port descriptor: target output port must be less than output count!");
    return m_output_port_descriptors[i];
}

std::shared_ptr<Node> Expression::get_node() const {
    if (!m_source_node)
        OPENVINO_THROW("An attempt to get uninitialized node from lowered expression");
    return  m_source_node;
}

std::shared_ptr<Emitter> Expression::get_emitter() const {
    return m_emitter;
}

RegInfo Expression::get_reg_info() const {
    RegInfo reg_info;
    reg_info.first.reserve(m_input_port_descriptors.size());
    reg_info.second.reserve(m_output_port_descriptors.size());
    for (const auto& port : m_input_port_descriptors)
        reg_info.first.push_back(port->get_reg());
    for (const auto& port : m_output_port_descriptors)
        reg_info.second.push_back(port->get_reg());
    return reg_info;
}

void Expression::set_reg_info(const RegInfo& rinfo) {
    const auto& in = rinfo.first;
    const auto& out = rinfo.second;
    OPENVINO_ASSERT(m_input_port_descriptors.size() == in.size(), "Incorrect count of input physical registers");
    OPENVINO_ASSERT(m_output_port_descriptors.size() == out.size(), "Incorrect count of output physical registers");
    for (size_t i = 0; i < m_input_port_descriptors.size(); ++i) {
        m_input_port_descriptors[i]->set_reg(in[i]);
    }
    for (size_t i = 0; i < m_output_port_descriptors.size(); ++i) {
        m_output_port_descriptors[i]->set_reg(out[i]);
    }
}

void Expression::validate() const {
    OPENVINO_ASSERT(m_source_node != nullptr,
                    "The expression has null source node");
    OPENVINO_ASSERT(m_input_port_descriptors.size() == m_input_port_connectors.size(),
                    "The count of input ports and input port connectors must be equal");
    OPENVINO_ASSERT(m_output_port_descriptors.size() == m_output_port_connectors.size(),
                    "The count of output ports and output port connectors must be equal");
}

void Expression::set_input_port_connector(size_t port, PortConnectorPtr to) {
    OPENVINO_ASSERT(port < get_input_count(), "Failed to set input PortConnector: target input port must be less than input count!");
    const auto& from = get_input_port_connector(port);
    if (from == to)
        return;

    const auto input_port = get_input_port(port);
    if (!to->found_consumer(input_port)) {
        to->add_consumer(input_port);
    }
    from->remove_consumer(input_port);
    // Set new PortConnector
    m_input_port_connectors[port] = std::move(to);
}

const std::vector<size_t>& Expression::get_loop_ids() const {
    return m_loop_ids;
}

void Expression::set_loop_ids(const std::vector<size_t>& loops) {
    std::unordered_set<size_t> s(loops.begin(), loops.end());
    OPENVINO_ASSERT(s.size() == loops.size(), "Loop IDs must be unique");
    m_loop_ids = loops;
}

ExpressionPtr Expression::clone_with_new_inputs(const std::shared_ptr<Node>& new_node,
                                                const std::vector<PortConnectorPtr>& new_inputs,
                                                const std::vector<PortDescriptorPtr>& new_in_descs) const {
    auto clone_ports_descriptors = [](const std::vector<PortDescriptorPtr>& src) {
        std::vector<PortDescriptorPtr> dst(src.size());
        for (size_t i = 0; i < src.size(); i++)
            dst[i] = src[i]->clone();
        return dst;
    };
    const auto& cloned = clone();
    OPENVINO_ASSERT(m_source_node->get_type_info() == new_node->get_type_info(),
                    "Can't clone expression for a new node with incompatible type");
    cloned->m_source_node = new_node;

    // Initialize Port Attributes: PortConnectors and PortDescriptors
    OPENVINO_ASSERT(new_in_descs.empty() || new_inputs.size() == new_in_descs.size(),
                    "Can't create Expression with new inputs: invalid number of input port connectors passed");
    cloned->m_input_port_descriptors = !new_in_descs.empty() ? new_in_descs : clone_ports_descriptors(m_input_port_descriptors);
    cloned->m_input_port_connectors = new_inputs;
    for (size_t i = 0; i < cloned->m_input_port_connectors.size(); i++) {
        const auto& i_con = cloned->m_input_port_connectors[i];
        const auto& i_port = cloned->get_input_port(i);
        if (!i_con->found_consumer(i_port))
            i_con->add_consumer(i_port);
    }
    cloned->m_output_port_descriptors = clone_ports_descriptors(m_output_port_descriptors);
    OPENVINO_ASSERT(cloned->m_output_port_connectors.size() == cloned->m_output_port_descriptors.size(),
                    "Can't create Expression with new inputs: output port attributes are not compatible");
    for (size_t i = 0; i < cloned->m_output_port_descriptors.size(); i++)
        cloned->m_output_port_connectors[i] = std::make_shared<PortConnector>(cloned->get_output_port(i));

    cloned->validate();
    return cloned;
}

ExpressionPtr Expression::clone_with_new_inputs(const ExpressionMap& expr_map,
                                                const std::shared_ptr<Node>& new_node) const {
    std::vector<PortConnectorPtr> new_inputs;
    new_inputs.reserve(m_input_port_connectors.size());
    for (const auto& input : m_input_port_connectors) {
        const auto& src_port = input->get_source();
        const auto& new_expr_it = expr_map.find(src_port.get_expr().get());
        if (new_expr_it != expr_map.end()) {
            const auto& new_expr = new_expr_it->second;
            new_inputs.emplace_back(new_expr->get_output_port_connector(src_port.get_index()));
        } else {
            new_inputs.emplace_back(input);
        }
    }
    return clone_with_new_inputs(new_node, new_inputs);
}

ExpressionPtr Expression::clone() const {
    return std::shared_ptr<Expression>(new Expression(*this));
}

bool Expression::visit_attributes(AttributeVisitor &visitor) {
    auto is_planar_layout = [](const std::vector<size_t>& layout) {
        for (size_t i = 0; i < layout.size(); ++i)
            if (layout[i] != i) return false;
        return true;
    };
    auto subtensor2str = [](const VectorDims& subtensor) {
        std::stringstream ss;
        for (size_t i = 0; i < subtensor.size(); ++i) {
            const auto& v = subtensor[i];
            const auto v_str = utils::is_full_dim_value(v) ? "FULL_DIM" :
                               utils::is_dynamic_value(v)  ? "?" : std::to_string(v);
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
    for (size_t i = 0; i < get_input_count(); i++) {
        const auto& desc = m_input_port_descriptors[i];
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
    for (size_t i = 0; i < get_output_count(); i++) {
        const auto& desc = m_output_port_descriptors[i];
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
    visitor.on_attribute("loop_ids", m_loop_ids);
    visitor.on_attribute("execution_number", m_exec_num);
    m_source_node->visit_attributes(visitor);
    return true;
}

ExpressionPort Expression::get_input_port(size_t i) {
    return ExpressionPort(shared_from_this(), ExpressionPort::Type::Input, i);
}

ExpressionPort Expression::get_output_port(size_t i) {
    return ExpressionPort(shared_from_this(), ExpressionPort::Type::Output, i);
}

std::vector<ExpressionPort> Expression::get_input_ports() {
    std::vector<ExpressionPort> ports;
    ports.reserve(get_input_count());
    for (size_t i = 0; i < get_input_count(); ++i) {
        ports.push_back(get_input_port(i));
    }
    return ports;
}

std::vector<ExpressionPort> Expression::get_output_ports() {
    std::vector<ExpressionPort> ports;
    ports.reserve(get_output_count());
    for (size_t i = 0; i < get_output_count(); ++i) {
        ports.push_back(get_output_port(i));
    }
    return ports;
}

void Expression::updateShapes() {
    OPENVINO_ASSERT(m_shapeInference, "Attempt to UpdateShapes without initialized shapeInference");
    IShapeInferSnippets::Result result;
    try {
        std::vector<VectorDimsRef> input_shapes;
        input_shapes.reserve(m_input_port_connectors.size());
        for (size_t i = 0; i < m_input_port_connectors.size(); i++) {
            const auto& src_port_desc = m_input_port_connectors[i]->get_source().get_descriptor_ptr();
            m_input_port_descriptors[i]->set_shape(src_port_desc->get_shape());
            // Note that input_shape is a reference, so we should always bind it to an object with a longer lifetime
            input_shapes.emplace_back(m_input_port_descriptors[i]->get_shape());
        }

        result = m_shapeInference->infer(input_shapes);
    }
    catch (const std::exception& exp) {
        OPENVINO_THROW("Shape inference of " + (m_source_node->get_friendly_name()) + " failed: " + exp.what());
    }
    OPENVINO_ASSERT(result.status == ShapeInferStatus::success,
                    "Shape inference of " + (m_source_node->get_friendly_name()) + " didn't return success status");
    OPENVINO_ASSERT(result.dims.size() == m_output_port_descriptors.size(), "shapeInference call returned invalid number of output shapes");
    for (size_t i = 0; i < m_output_port_descriptors.size(); i++)
        m_output_port_descriptors[i]->set_shape(result.dims[i]);
}

} // namespace lowered
} // namespace snippets
} // namespace ov
