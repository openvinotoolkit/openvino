// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "openvino/reference/utils/registers_pool.hpp"

namespace ov {
namespace reference {
namespace jit {

RegistersPool::RegistersPool(int simd_registers_number) : m_general_set(16), m_simd_set(simd_registers_number) {
    check_unique_and_update();
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RAX));
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RCX));
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RDI));
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RBP));
}

RegistersPool::RegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude, int simd_registers_number)
    : m_general_set(16),
      m_simd_set(simd_registers_number) {
    check_unique_and_update();
    for (auto& reg : regsToExclude) {
        if (reg.isXMM() || reg.isYMM() || reg.isZMM()) {
            m_simd_set.exclude(reg);
        } else if (reg.isREG()) {
            m_general_set.exclude(reg);
        }
    }
    m_general_set.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
}

void RegistersPool::check_unique_and_update(bool is_ctor) {
    static thread_local bool is_created = false;
    if (is_ctor) {
        if (is_created) {
            OPENVINO_THROW("There should be only one instance of RegistersPool per thread");
        }
        is_created = true;
    } else {
        is_created = false;
    }
}

void RegistersPool::PhysicalSet::set_as_used(size_t reg_idx) {
    if (reg_idx >= m_is_free_index_vector.size()) {
        OPENVINO_THROW("reg_idx is out of bounds in RegistersPool::PhysicalSet::set_as_used()");
    }
    if (!m_is_free_index_vector[reg_idx]) {
        OPENVINO_THROW("Inconsistency in RegistersPool::PhysicalSet::set_as_used()");
    }
    m_is_free_index_vector[reg_idx] = false;
}

void RegistersPool::PhysicalSet::set_as_unused(size_t reg_idx) {
    if (reg_idx >= m_is_free_index_vector.size()) {
        OPENVINO_THROW("reg_idx is out of bounds in RegistersPool::PhysicalSet::set_as_used()");
    }
    if (m_is_free_index_vector[reg_idx]) {
        OPENVINO_THROW("Inconsistency in RegistersPool::PhysicalSet::set_as_unused()");
    }
    m_is_free_index_vector[reg_idx] = true;
}

size_t RegistersPool::PhysicalSet::get_unused(size_t requested_idx) {
    if (requested_idx == static_cast<size_t>(any_idx)) {
        return get_first_free_index();
    } else {
        if (requested_idx >= m_is_free_index_vector.size()) {
            OPENVINO_THROW("requested_idx is out of bounds in RegistersPool::PhysicalSet::get_unused()");
        }
        if (!m_is_free_index_vector[requested_idx]) {
            OPENVINO_THROW("The register with index #", requested_idx, " already used in the RegistersPool");
        }
        return requested_idx;
    }
}

size_t RegistersPool::PhysicalSet::count_unused() const {
    size_t count = 0;
    for (const auto& isFree : m_is_free_index_vector) {
        if (isFree) {
            ++count;
        }
    }
    return count;
}

size_t RegistersPool::PhysicalSet::get_first_free_index() {
    for (size_t c = 0; c < m_is_free_index_vector.size(); ++c) {
        if (m_is_free_index_vector[c]) {
            return c;
        }
    }
    OPENVINO_THROW("Not enough registers in the RegistersPool");
}

}  // namespace jit
}  // namespace reference
}  // namespace ov

#endif  // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64
