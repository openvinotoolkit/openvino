// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

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

    /**
    * @interface PortDescriptor
    * @brief This class describes port of MemoryAccess operation
    * @param m_count - count of elements to load/store
    * @param m_offset - starting index of elements to load/store
    * @param m_index - port index
    * @ingroup snippets
    */
    struct PortDescriptor {
        PortDescriptor(size_t count, size_t offset) : count(count), offset(offset) {}
        PortDescriptor() = default;

        size_t count = 0lu;
        size_t offset = 0lu;
        size_t index = 0lu;

    private:
        PortDescriptor(size_t count, size_t offset, size_t index) : count(count), offset(offset), index(index) {}

        friend class MemoryAccess;
    };

    void set_input_count(size_t count, size_t idx = 0);
    void set_output_count(size_t count, size_t idx = 0);
    void set_input_offset(size_t offset, size_t idx = 0);
    void set_output_offset(size_t offset, size_t idx = 0);

    size_t get_input_count(size_t idx = 0) const;
    size_t get_output_count(size_t idx = 0) const;
    size_t get_input_offset(size_t idx = 0) const;
    size_t get_output_offset(size_t idx = 0) const;

    size_t get_input_port_count() const { return m_input_ports.size(); }
    size_t get_output_port_count() const { return m_output_ports.size(); }

    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    explicit MemoryAccess(const OutputVector& arguments, size_t input_count = 0, size_t output_count = 0);
    MemoryAccess() = default;

    void set_input_port_descriptor(const PortDescriptor& desc, const size_t i);
    void set_output_port_descriptor(const PortDescriptor& desc, const size_t i);
    const PortDescriptor& get_input_port_descriptor(const size_t i) const;
    const PortDescriptor& get_output_port_descriptor(const size_t i) const;

    std::vector<PortDescriptor> m_input_ports;
    std::vector<PortDescriptor> m_output_ports;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
