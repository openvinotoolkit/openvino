// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

class MemoryAccess;

/**
* @interface PortDescriptor
* @brief This class describes port of MemoryAccess operation
* @param m_count - count of elements to load/store
* @param m_offset - starting index of elements to load/store
* @param m_index - port index
* @ingroup snippets
*/

struct PortDescriptor {
    PortDescriptor(size_t count, size_t offset) : m_count(count), m_offset(offset) {}
    PortDescriptor() = default;

    size_t m_count = 0lu;
    size_t m_offset = 0lu;
    size_t m_index = 0lu;

private:
    PortDescriptor(size_t count, size_t offset, size_t index) : m_count(count), m_offset(offset), m_index(index) {}

    friend class MemoryAccess;
};

/**
 * @interface MemoryAccess
 * @brief This is a base class for memory access operations (like Load and Store).
 *        It provides universal interface to manipulate with memory: load/store.
 * @param m_input_ports - vector of input descriptors: variables of PortDescriptor class
 * @param m_output_ports - vector of output descriptors: variables of PortDescriptor class
 * @ingroup snippets
 */

class MemoryAccess : public ngraph::op::Op {
public:
    OPENVINO_OP("MemoryAccess", "SnippetsOpset");

    void set_input_port_descriptor(const PortDescriptor& desc, const size_t i);
    void set_output_port_descriptor(const PortDescriptor& desc, const size_t i);
    PortDescriptor get_input_port_descriptor(const size_t i) const;
    PortDescriptor get_output_port_descriptor(const size_t i) const;
    PortDescriptor& get_input_port_descriptor(const size_t i);
    PortDescriptor& get_output_port_descriptor(const size_t i);

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    explicit MemoryAccess(const OutputVector& arguments);
    MemoryAccess() = default;

    std::vector<PortDescriptor> m_input_ports;
    std::vector<PortDescriptor> m_output_ports;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
