// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "snippets/op/memory_access.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace modifier {

/**
 * @interface TensorProcPrim
 * @brief TensorProcPrim is modifier to mark operations supported with TPP
 * @ingroup snippets
 */
class TensorProcessingPrimitive : virtual public snippets::modifier::MemoryAccess {
    public:
        void clone_memory_acess_ports(const TensorProcessingPrimitive& other) {
            m_input_ports = other.m_input_ports;
            m_output_ports = other.m_output_ports;
        }
};

} // namespace modifier
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
