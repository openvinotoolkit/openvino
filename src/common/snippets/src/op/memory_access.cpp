// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/memory_access.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace modifier {

MemoryAccess::MemoryAccess(size_t input_count, size_t output_count) {
    auto init_iota_set = [](size_t num) {
        if (num == 0)
            return std::set<size_t>{};
        std::vector<size_t> vec(num);
        std::iota(vec.begin(), vec.end(), 0);
        return std::set<size_t>(vec.begin(), vec.end());
    };
    ctor_initialize(init_iota_set(input_count), init_iota_set(output_count));
}

MemoryAccess::MemoryAccess(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) {
    ctor_initialize(input_ports, output_ports);
}

MemoryAccess::MemoryAccess(const PortMap& input_ports, const PortMap& output_ports)
    : m_input_ports(input_ports), m_output_ports(output_ports) {}

void MemoryAccess::ctor_initialize(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports) {
    for (auto port : input_ports) {
        m_input_ports[port] = {0, 0, 0, port};
    }
    for (auto port : output_ports) {
        m_output_ports[port] = {0, 0, 0, port};
    }
}

bool MemoryAccess::is_full_memory_access_op(const std::shared_ptr<ov::Node>& op) const {
    for (size_t i = 0; i < op->get_input_size(); ++i) {
        if (!is_memory_access_input_port(i))
            return false;
    }
    for (size_t i = 0; i < op->get_output_size(); ++i) {
        if (!is_memory_access_output_port(i))
            return false;
    }
    return true;
}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    bool is_dynamic = false;
    for (const auto& p : m_input_ports) {
        auto idx = p.first;
        auto port = p.second;
        auto count = utils::value2str(port.count);
        auto offset = utils::value2str(port.offset);
        auto stride = utils::value2str(port.stride);
        visitor.on_attribute("count_in_" + std::to_string(idx), count);
        visitor.on_attribute("offset_in_" + std::to_string(idx), offset);
        visitor.on_attribute("stride_in_" + std::to_string(idx), stride);
        is_dynamic |= utils::is_dynamic_value(port.count) || utils::is_dynamic_value(port.offset) || utils::is_dynamic_value(port.stride);
    }
    for (const auto& p : m_output_ports) {
        auto idx = p.first;
        auto port = p.second;
        auto count = utils::value2str(port.count);
        auto offset = utils::value2str(port.offset);
        auto stride = utils::value2str(port.stride);
        visitor.on_attribute("count_out_" + std::to_string(idx), count);
        visitor.on_attribute("offset_out_" + std::to_string(idx), offset);
        visitor.on_attribute("stride_out_" + std::to_string(idx), stride);
        is_dynamic |= utils::is_dynamic_value(port.count) || utils::is_dynamic_value(port.offset) || utils::is_dynamic_value(port.stride);
    }

    std::string dynamic_status = is_dynamic ? "DYNAMIC" : "STATIC";
    visitor.on_attribute("dynamic_status", dynamic_status);

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
    it->second = desc;
    it->second.index = i;
}

void MemoryAccess::set_output_port_descriptor(const PortDescriptor& desc, const size_t i) {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    it->second = desc;
    it->second.index = i;
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) const {
    const auto it = m_input_ports.find(i);
    OPENVINO_ASSERT(it != m_input_ports.end(), "Index of input port descriptor should be less than count of input ports");
    return it->second;
}

MemoryAccess::PortDescriptor& MemoryAccess::get_input_port_descriptor(const size_t i) {
    return const_cast<MemoryAccess::PortDescriptor&>(const_cast<const MemoryAccess*>(this)->get_input_port_descriptor(i));
}

const MemoryAccess::PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) const {
    const auto it = m_output_ports.find(i);
    OPENVINO_ASSERT(it != m_output_ports.end(), "Index of output port descriptor should be less than count of output ports");
    return it->second;
}

MemoryAccess::PortDescriptor& MemoryAccess::get_output_port_descriptor(const size_t i) {
    return const_cast<MemoryAccess::PortDescriptor&>(const_cast<const MemoryAccess*>(this)->get_output_port_descriptor(i));
}

void  MemoryAccess::set_input_count(size_t count, size_t idx) {
    get_input_port_descriptor(idx).count = count;
}
void MemoryAccess::set_output_count(size_t count, size_t idx) {
    get_output_port_descriptor(idx).count = count;
}
void  MemoryAccess::set_input_offset(size_t offset, size_t idx) {
    get_input_port_descriptor(idx).offset = offset;
}
void MemoryAccess::set_output_offset(size_t offset, size_t idx) {
    get_output_port_descriptor(idx).offset = offset;
}
void  MemoryAccess::set_input_stride(size_t stride, size_t idx) {
    get_input_port_descriptor(idx).stride = stride;
}
void  MemoryAccess::set_output_stride(size_t stride, size_t idx) {
    get_output_port_descriptor(idx).stride = stride;
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
size_t MemoryAccess::get_input_stride(size_t idx) const {
    return get_input_port_descriptor(idx).stride;
}
size_t MemoryAccess::get_output_stride(size_t idx) const {
    return get_output_port_descriptor(idx).stride;
}

} // namespace modifier
} // namespace snippets
} // namespace ov
