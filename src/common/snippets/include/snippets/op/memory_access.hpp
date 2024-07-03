// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace modifier {

/**
 * @interface MemoryAccess
 * @brief This is a base class for memory access operations (like Load and Store).
 *        It provides universal interface to manipulate with memory: load/store.
 * @param m_input_ports - map of input descriptors: variables of PortDescriptor class
 * @param m_output_ports - map of output descriptors: variables of PortDescriptor class
 * @ingroup snippets
 */

class MemoryAccess {
public:
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
        // TODO: should we deprecate count in favor of subtensors, ticket: 130004
        size_t count = 0lu;
        size_t offset = 0lu;
        // Note: stride is interpreted as leading dimension for 2D subtensor ops
        size_t stride = 0lu;
        size_t index = 0lu;

    private:
        PortDescriptor(size_t count, size_t offset, size_t stride, size_t index) :
            count(count), offset(offset), stride(stride), index(index) {}

        friend class MemoryAccess;
    };
    using PortMap = std::map<size_t, PortDescriptor>;

    void set_input_count(size_t count, size_t idx = 0);
    void set_output_count(size_t count, size_t idx = 0);
    void set_input_offset(size_t offset, size_t idx = 0);
    void set_output_offset(size_t offset, size_t idx = 0);
    void set_input_stride(size_t stride, size_t idx = 0);
    void set_output_stride(size_t stride, size_t idx = 0);

    size_t get_input_count(size_t idx = 0) const;
    size_t get_output_count(size_t idx = 0) const;
    size_t get_input_offset(size_t idx = 0) const;
    size_t get_output_offset(size_t idx = 0) const;
    size_t get_input_stride(size_t idx = 0) const;
    size_t get_output_stride(size_t idx = 0) const;

    PortMap get_memory_access_input_ports() const { return m_input_ports; }
    PortMap get_memory_access_output_ports() const { return m_output_ports; }

    bool is_memory_access_input_port(size_t idx) const;
    bool is_memory_access_output_port(size_t idx) const;

    /**
     * @brief Checks if the provided operation memory access on all ports
     */
    bool is_full_memory_access_op(const std::shared_ptr<ov::Node>& op) const;

    bool visit_attributes(AttributeVisitor& visitor);

protected:
    explicit MemoryAccess(size_t input_count, size_t output_count = 0);
    explicit MemoryAccess(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports);
    explicit MemoryAccess(const PortMap& input_ports, const PortMap& output_ports);
    MemoryAccess() = default;

    // This method can be called only in ctors
    void ctor_initialize(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports);

    void set_input_port_descriptor(const PortDescriptor& desc, const size_t i);
    void set_output_port_descriptor(const PortDescriptor& desc, const size_t i);
    const PortDescriptor& get_input_port_descriptor(const size_t i) const;
    const PortDescriptor& get_output_port_descriptor(const size_t i) const;
    PortDescriptor& get_input_port_descriptor(const size_t i);
    PortDescriptor& get_output_port_descriptor(const size_t i);

    // [port_num, port_desc]
    PortMap m_input_ports;
    PortMap m_output_ports;
};

} // namespace modifier
} // namespace snippets
} // namespace ov
