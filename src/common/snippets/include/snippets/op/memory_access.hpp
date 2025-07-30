// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <set>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

namespace ov::snippets::modifier {

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
        size_t count = 0LU;
        size_t offset = 0LU;
        // Note: stride is interpreted as leading dimension for 2D subtensor ops
        size_t stride = 0LU;
        size_t index = 0LU;

    private:
        PortDescriptor(size_t count, size_t offset, size_t stride, size_t index)
            : count(count),
              offset(offset),
              stride(stride),
              index(index) {}

        friend class MemoryAccess;
    };
    using PortMap = std::map<size_t, PortDescriptor>;

    void set_input_count(size_t count, size_t idx = 0);
    void set_output_count(size_t count, size_t idx = 0);
    void set_input_offset(size_t offset, size_t idx = 0);
    void set_output_offset(size_t offset, size_t idx = 0);
    void set_input_stride(size_t stride, size_t idx = 0);
    void set_output_stride(size_t stride, size_t idx = 0);

    [[nodiscard]] size_t get_input_count(size_t idx = 0) const;
    [[nodiscard]] size_t get_output_count(size_t idx = 0) const;
    [[nodiscard]] size_t get_input_offset(size_t idx = 0) const;
    [[nodiscard]] size_t get_output_offset(size_t idx = 0) const;
    [[nodiscard]] size_t get_input_stride(size_t idx = 0) const;
    [[nodiscard]] size_t get_output_stride(size_t idx = 0) const;

    [[nodiscard]] PortMap get_memory_access_input_ports() const {
        return m_input_ports;
    }
    [[nodiscard]] PortMap get_memory_access_output_ports() const {
        return m_output_ports;
    }

    [[nodiscard]] bool is_memory_access_input_port(size_t idx) const;
    [[nodiscard]] bool is_memory_access_output_port(size_t idx) const;

    /**
     * @brief Checks if the provided operation memory access on all ports
     */
    [[nodiscard]] bool is_full_memory_access_op(const std::shared_ptr<ov::Node>& op) const;

    bool visit_attributes(AttributeVisitor& visitor);

protected:
    explicit MemoryAccess(size_t input_count, size_t output_count = 0);
    explicit MemoryAccess(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports);
    explicit MemoryAccess(PortMap input_ports, PortMap output_ports);
    MemoryAccess() = default;

    // This method can be called only in ctors
    void ctor_initialize(const std::set<size_t>& input_ports, const std::set<size_t>& output_ports);

    void set_input_port_descriptor(const PortDescriptor& desc, size_t i);
    void set_output_port_descriptor(const PortDescriptor& desc, size_t i);
    [[nodiscard]] const PortDescriptor& get_input_port_descriptor(size_t i) const;
    [[nodiscard]] const PortDescriptor& get_output_port_descriptor(size_t i) const;
    PortDescriptor& get_input_port_descriptor(size_t i);
    PortDescriptor& get_output_port_descriptor(size_t i);

    // [port_num, port_desc]
    PortMap m_input_ports;
    PortMap m_output_ports;
};

}  // namespace ov::snippets::modifier
