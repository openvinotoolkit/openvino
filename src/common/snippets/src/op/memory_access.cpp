// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/op/memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const OutputVector& arguments) : Op(arguments) {}

void MemoryAccess::validate_and_infer_types() {
    // We create descriptors in validate_and_infer_types() (instead of in ctor)
    const auto input_count = get_input_size();
    const auto output_count = get_output_size();
    while (m_input_ports.size() < input_count) {
        m_input_ports.push_back({0, 0, m_input_ports.size()});
    }
    while (m_output_ports.size() < output_count) {
        m_output_ports.push_back({0, 0, m_output_ports.size()});
    }
    OPENVINO_ASSERT(m_input_ports.size() == input_count, "The count of input ports must be equal to input count");
    OPENVINO_ASSERT(m_output_ports.size() == output_count, "The count of output ports must be equal to output count");
}

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
    NGRAPH_CHECK(i < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[i] = { desc.m_count, desc.m_offset, i};
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    // Logic is as same as ov::Node::get_output_descriptor
    NGRAPH_CHECK(i < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[i] = { desc.m_count, desc.m_offset, i};
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
    NGRAPH_CHECK(idx < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[idx].m_count = count;
}
void MemoryAccess::set_output_count(size_t count, size_t idx) {
    NGRAPH_CHECK(idx < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[idx].m_count = count;
}
void  MemoryAccess::set_input_offset(size_t offset, size_t idx) {
    NGRAPH_CHECK(idx < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    m_input_ports[idx].m_offset = offset;
}
void MemoryAccess::set_output_offset(size_t offset, size_t idx) {
    NGRAPH_CHECK(idx < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    m_output_ports[idx].m_offset = offset;
}
size_t MemoryAccess::get_input_count(size_t idx) const {
    NGRAPH_CHECK(idx < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    return m_input_ports[idx].m_count;
}
size_t MemoryAccess::get_output_count(size_t idx) const {
    NGRAPH_CHECK(idx < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    return m_output_ports[idx].m_count;
}
size_t MemoryAccess::get_input_offset(size_t idx) const {
    NGRAPH_CHECK(idx < m_input_ports.size(), "Index of input port descriptor should be less than count of input ports");
    return m_input_ports[idx].m_offset;
}
size_t MemoryAccess::get_output_offset(size_t idx) const {
    NGRAPH_CHECK(idx < m_output_ports.size(), "Index of output port descriptor should be less than count of output ports");
    return m_output_ports[idx].m_offset;
}

} // namespace op
} // namespace snippets
} // namespace ngraph
