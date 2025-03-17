// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "snippets/op/memory_access.hpp"

namespace ov::intel_cpu::tpp::modifier {

/**
 * @interface TensorProcessingPrimitive
 * @brief TensorProcessingPrimitive a is modifier to mark operations supported with TPP
 * @ingroup snippets
 */
class TensorProcessingPrimitive : virtual public snippets::modifier::MemoryAccess {
public:
    void clone_memory_access_ports(const TensorProcessingPrimitive& other) {
        m_input_ports = other.m_input_ports;
        m_output_ports = other.m_output_ports;
    }
    static bool visit_attributes(AttributeVisitor& visitor) {
        std::string modifier{"TPP"};
        visitor.on_attribute("modifier", modifier);
        return true;
    }
};

}  // namespace ov::intel_cpu::tpp::modifier
