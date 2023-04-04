// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/op/memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const OutputVector& arguments, size_t input_count, size_t output_count) : Op(arguments) {
    while (m_input_ports.size() < input_count) {
        m_input_ports.push_back({0, 0, m_input_ports.size()});
    }
    while (m_output_ports.size() < output_count) {
        m_output_ports.push_back({0, 0, m_output_ports.size()});
    }
}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    for (size_t i = 0; i < m_input_ports.size(); ++i) {
        auto port = m_input_ports[i];
        visitor.on_attribute("count_in_" + std::to_string(i), port.count);
        visitor.on_attribute("offset_in_" + std::to_string(i), port.offset);
    }
    for (size_t i = 0; i < m_output_ports.size(); ++i) {
        auto port = m_output_ports[i];
        visitor.on_attribute("count_out_" + std::to_string(i), port.count);
        visitor.on_attribute("offset_out_" + std::to_string(i), port.offset);
    }
    return true;
}

void MemoryAccess::set_input_port_descriptor(const PortDescriptor& desc, const size_t i) {
    NGRAPH_CHECK(i < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[i] = { desc.count, desc.offset, i};
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    NGRAPH_CHECK(i < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[i] = { desc.count, desc.offset, i};
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) const {
    NGRAPH_CHECK(i < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    return m_input_ports[i];
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) const {
    NGRAPH_CHECK(i < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    return m_output_ports[i];
}

void  MemoryAccess::set_input_count(size_t count, size_t idx) {
    set_input_port_descriptor({count, get_input_port_descriptor(idx).offset, idx}, idx);
}
void MemoryAccess::set_output_count(size_t count, size_t idx) {
    set_output_port_descriptor({count, get_output_port_descriptor(idx).offset, idx}, idx);
}
void  MemoryAccess::set_input_offset(size_t offset, size_t idx) {
    set_input_port_descriptor({get_input_port_descriptor(idx).count, offset, idx}, idx);
}
void MemoryAccess::set_output_offset(size_t offset, size_t idx) {
    set_output_port_descriptor({get_output_port_descriptor(idx).count, offset, idx}, idx);
}
size_t MemoryAccess::get_input_count(size_t idx) const {
    return get_input_port_descriptor(idx).count;
}
size_t MemoryAccess::get_output_count(size_t idx) const {
    return get_output_port_descriptor(idx).count;
}
size_t MemoryAccess::get_input_offset(size_t idx) const {
    return get_input_port_descriptor(idx).offset;
}
size_t MemoryAccess::get_output_offset(size_t idx) const {
    return get_output_port_descriptor(idx).offset;
}

} // namespace op
} // namespace snippets
} // namespace ngraph
