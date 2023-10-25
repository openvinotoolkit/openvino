// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const OutputVector& arguments, size_t input_count, size_t output_count) : Op(arguments) {
    auto init_iota_set = [](size_t num) {
        if (num == 0)
            return std::set<size_t>{};
        std::vector<size_t> vec(num);
        std::iota(vec.begin(), vec.end(), 0);
        return std::set<size_t>(vec.begin(), vec.end());
    };
    ctor_initialize(init_iota_set(input_count), init_iota_set(output_count));
}

MemoryAccess::MemoryAccess(const OutputVector& arguments, const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) : Op(arguments) {
    ctor_initialize(input_ports, output_ports);
}

MemoryAccess::MemoryAccess(const OutputVector& arguments, const PortMap& input_ports, const PortMap& output_ports)
    : Op(arguments), m_input_ports(input_ports), m_output_ports(output_ports) {}

void MemoryAccess::ctor_initialize(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) {
    for (auto port : input_ports) {
        m_input_ports[port] = {0, 0, port};
    }
    for (auto port : output_ports) {
        m_output_ports[port] = {0, 0, port};
    }
}

bool MemoryAccess::is_full_memory_access_op() const {
    for (size_t i = 0; i < get_input_size(); ++i) {
        if (!is_memory_access_input_port(i))
            return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i) {
        if (!is_memory_access_output_port(i))
            return false;
    }
    return true;
}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    for (const auto& p : m_input_ports) {
        auto idx = p.first;
        auto port = p.second;
        visitor.on_attribute("count_in_" + std::to_string(idx), port.count);
        visitor.on_attribute("offset_in_" + std::to_string(idx), port.offset);
    }
    for (const auto& p : m_output_ports) {
        auto idx = p.first;
        auto port = p.second;
        visitor.on_attribute("count_out_" + std::to_string(idx), port.count);
        visitor.on_attribute("offset_out_" + std::to_string(idx), port.offset);
    }
    return true;
}

bool MemoryAccess::is_memory_access_input_port(size_t idx) const {
    return m_input_ports.find(idx) != m_input_ports.end();
}
bool MemoryAccess::is_memory_access_output_port(size_t idx) const {
    return m_output_ports.find(idx) != m_output_ports.end();
}

void MemoryAccess::set_input_port_descriptor(const PortDescriptor& desc, const size_t i) {
    const auto it = m_input_ports.find(i);
    OPENVINO_ASSERT(it != m_input_ports.end(), "Index of input port descriptor should be less than count of input ports");
    (*it).second = { desc.count, desc.offset, i};
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    (*it).second = { desc.count, desc.offset, i};
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) const {
    const auto it = m_input_ports.find(i);
    OPENVINO_ASSERT(it != m_input_ports.end(), "Index of input port descriptor should be less than count of input ports");
    return (*it).second;
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) const {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    return (*it).second;
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
} // namespace ov
