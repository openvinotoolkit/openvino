// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/memory_access.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const OutputVector& arguments) : Op(arguments) {}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    for (size_t i = 0; i < m_input_ports.size(); ++i) {
        auto port = m_input_ports[i];
        visitor.on_attribute("count_in_" + std::to_string(i), port.m_count);
        visitor.on_attribute("offset_in_" + std::to_string(i), port.m_offset);
    }
    for (size_t i = 0; i < m_output_ports.size(); ++i) {
        auto port = m_output_ports[i];
        visitor.on_attribute("count_out_" + std::to_string(i), port.m_count);
        visitor.on_attribute("offset_out_" + std::to_string(i), port.m_offset);
    }
    return true;
}

void MemoryAccess::set_input_port_descriptor(const PortDescriptor& desc, const size_t i) {
    // Logic is as same as ov::Node::get_input_descriptor
    while (m_input_ports.size() <= i) {
        m_input_ports.emplace_back(PortDescriptor{0, 0, m_input_ports.size()});
    }
    m_input_ports[i] = { desc.m_count, desc.m_offset, i};
}

PortDescriptor MemoryAccess::get_input_port_descriptor(const size_t i) const {
    // We cannot use the same way as in ov::Node::get_input_descriptor because this method must be const
    // to allow call const Derived::clone_with_new_inputs() method
    NGRAPH_CHECK(i < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    return m_input_ports[i];
}

PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) {
    // Logic is as same as ov::Node::get_input_descriptor
    while (m_input_ports.size() <= i) {
        m_input_ports.emplace_back(PortDescriptor{0, 0, m_input_ports.size()});
    }
    return m_input_ports[i];
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    // Logic is as same as ov::Node::get_output_descriptor
    while (m_output_ports.size() <= i) {
        m_output_ports.emplace_back(PortDescriptor{0, 0, m_output_ports.size()});
    }
    m_output_ports[i] = { desc.m_count, desc.m_offset, i};
}

PortDescriptor MemoryAccess::get_output_port_descriptor(const size_t i) const {
    // We cannot use the same way as in ov::Node::get_input_descriptor because this method must be const
    // to allow call const Derived::clone_with_new_inputs() method
    NGRAPH_CHECK(i < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    return m_output_ports[i];
}

PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) {
    // Logic is as same as ov::Node::get_output_descriptor
    while (m_output_ports.size() <= i) {
        m_output_ports.emplace_back(PortDescriptor{0, 0, m_output_ports.size()});
    }
    return m_output_ports[i];
}

} // namespace op
} // namespace snippets
} // namespace ngraph