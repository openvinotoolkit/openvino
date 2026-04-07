// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_reg_spill_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/reg_spill.hpp"

namespace ov::intel_cpu::aarch64 {

class EmitABIRegSpills {
public:
    explicit EmitABIRegSpills(dnnl::impl::cpu::aarch64::jit_generator_t* h_arg) : h(h_arg) {}

    ~EmitABIRegSpills() {
        OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
    }

    [[nodiscard]] size_t get_num_spilled_regs() const {
        return m_gpr_regs_to_spill.size() + m_vec_regs_to_spill.size();
    }

    void preamble(const std::set<snippets::Reg>& live_regs) {
        OPENVINO_ASSERT(spill_status, "Attempt to spill ABI registers twice in a row");

        reset();
        for (const auto& reg : live_regs) {
            switch (reg.type) {
            case snippets::RegType::gpr:
                m_gpr_regs_to_spill.push_back(reg.idx);
                break;
            case snippets::RegType::vec:
                m_vec_regs_to_spill.push_back(reg.idx);
                break;
            default:
                OPENVINO_THROW("Unsupported register type in Arm64 RegSpill emitter");
            }
        }

        m_total_gpr_shift = aligned_size(static_cast<uint32_t>(m_gpr_regs_to_spill.size() * gpr_size));
        m_total_vec_shift = aligned_size(static_cast<uint32_t>(m_vec_regs_to_spill.size() * vec_size));
        m_total_shift = m_total_gpr_shift + m_total_vec_shift;

        if (m_total_shift > 0) {
            h->sub(h->sp, h->sp, m_total_shift);
        }

        store_gprs();
        store_vecs();
        spill_status = false;
    }

    void postamble() {
        OPENVINO_ASSERT(!spill_status, "Attempt to restore ABI registers that were not spilled");

        restore_vecs();
        restore_gprs();

        if (m_total_shift > 0) {
            h->add(h->sp, h->sp, m_total_shift);
        }

        reset();
        spill_status = true;
    }

private:
    static constexpr uint32_t stack_alignment = jit_emitter::sp_alignment;
    static constexpr uint32_t gpr_size = 8;
    static constexpr uint32_t vec_size = 16;

    [[nodiscard]] static uint32_t aligned_size(uint32_t size) {
        if (size == 0) {
            return 0;
        }
        return ((size + stack_alignment - 1) / stack_alignment) * stack_alignment;
    }

    void store_gprs() const {
        int32_t current_offset = 0;
        size_t i = 0;
        for (; i + 1 < m_gpr_regs_to_spill.size(); i += 2) {
            h->stp(Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i]),
                   Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i + 1]),
                   Xbyak_aarch64::ptr(h->sp, current_offset));
            current_offset += static_cast<int32_t>(2 * gpr_size);
        }
        if (i < m_gpr_regs_to_spill.size()) {
            h->str(Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
        }
    }

    void store_vecs() const {
        auto current_offset = static_cast<int32_t>(m_total_gpr_shift);
        size_t i = 0;
        for (; i + 1 < m_vec_regs_to_spill.size(); i += 2) {
            h->stp(Xbyak_aarch64::QReg(m_vec_regs_to_spill[i]),
                   Xbyak_aarch64::QReg(m_vec_regs_to_spill[i + 1]),
                   Xbyak_aarch64::ptr(h->sp, current_offset));
            current_offset += static_cast<int32_t>(2 * vec_size);
        }
        if (i < m_vec_regs_to_spill.size()) {
            h->str(Xbyak_aarch64::QReg(m_vec_regs_to_spill[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
        }
    }

    void restore_vecs() const {
        auto current_offset = static_cast<int32_t>(m_total_gpr_shift);
        size_t i = 0;
        for (; i + 1 < m_vec_regs_to_spill.size(); i += 2) {
            h->ldp(Xbyak_aarch64::QReg(m_vec_regs_to_spill[i]),
                   Xbyak_aarch64::QReg(m_vec_regs_to_spill[i + 1]),
                   Xbyak_aarch64::ptr(h->sp, current_offset));
            current_offset += static_cast<int32_t>(2 * vec_size);
        }
        if (i < m_vec_regs_to_spill.size()) {
            h->ldr(Xbyak_aarch64::QReg(m_vec_regs_to_spill[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
        }
    }

    void restore_gprs() const {
        int32_t current_offset = 0;
        size_t i = 0;
        for (; i + 1 < m_gpr_regs_to_spill.size(); i += 2) {
            h->ldp(Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i]),
                   Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i + 1]),
                   Xbyak_aarch64::ptr(h->sp, current_offset));
            current_offset += static_cast<int32_t>(2 * gpr_size);
        }
        if (i < m_gpr_regs_to_spill.size()) {
            h->ldr(Xbyak_aarch64::XReg(m_gpr_regs_to_spill[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
        }
    }

    void reset() {
        m_gpr_regs_to_spill.clear();
        m_vec_regs_to_spill.clear();
        m_total_gpr_shift = 0;
        m_total_vec_shift = 0;
        m_total_shift = 0;
    }

    dnnl::impl::cpu::aarch64::jit_generator_t* h = nullptr;
    std::vector<size_t> m_gpr_regs_to_spill;
    std::vector<size_t> m_vec_regs_to_spill;
    uint32_t m_total_gpr_shift = 0;
    uint32_t m_total_vec_shift = 0;
    uint32_t m_total_shift = 0;
    bool spill_status = true;
};

/* ================== jit_reg_spill_begin_emitters ====================== */

jit_reg_spill_begin_emitter::jit_reg_spill_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                                         const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    const auto& reg_spill_node = ov::as_type_ptr<snippets::op::RegSpillBegin>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(reg_spill_node, "expects RegSpillBegin expression");
    const auto& rinfo = expr->get_reg_info();
    m_regs_to_spill = std::set<snippets::Reg>(rinfo.second.begin(), rinfo.second.end());
    m_abi_reg_spiller = std::make_shared<EmitABIRegSpills>(h);
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_reg_spill_begin_emitter::validate_arguments(const std::vector<size_t>& in,
                                                     const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "In regs should be empty for reg_spill_begin emitter");
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == m_regs_to_spill.size(),
                              "Invalid number of out regs for reg_spill_begin emitter");
}

void jit_reg_spill_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                                 const std::vector<size_t>& out,
                                                 [[maybe_unused]] const std::vector<size_t>& pool_vec_idxs,
                                                 [[maybe_unused]] const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                            [[maybe_unused]] const std::vector<size_t>& out) const {
    m_abi_reg_spiller->preamble(m_regs_to_spill);
}

/* ============================================================== */

/* ================== jit_reg_spill_end_emitter ====================== */

jit_reg_spill_end_emitter::jit_reg_spill_end_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                                     dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    OV_CPU_JIT_EMITTER_ASSERT(ov::is_type<snippets::op::RegSpillEnd>(expr->get_node()) && expr->get_input_count() > 0,
                              "Invalid expression in RegSpillEnd emitter");

    const auto& parent_expr = expr->get_input_expr_ptr(0);
    const auto& reg_spill_begin_emitter =
        std::dynamic_pointer_cast<jit_reg_spill_begin_emitter>(parent_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(reg_spill_begin_emitter, "Failed to obtain reg_spill_begin emitter");
    m_abi_reg_spiller = reg_spill_begin_emitter->m_abi_reg_spiller;
}

void jit_reg_spill_end_emitter::validate_arguments(const std::vector<size_t>& in,
                                                   const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(out.empty(), "Out regs should be empty for reg_spill_end emitter");
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == m_abi_reg_spiller->get_num_spilled_regs(),
                              "Invalid number of in regs for reg_spill_end emitter");
}

void jit_reg_spill_end_emitter::emit_code_impl(const std::vector<size_t>& in,
                                               const std::vector<size_t>& out,
                                               [[maybe_unused]] const std::vector<size_t>& pool_vec_idxs,
                                               [[maybe_unused]] const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_reg_spill_end_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                          [[maybe_unused]] const std::vector<size_t>& out) const {
    m_abi_reg_spiller->postamble();
}

/* ============================================================== */

}  // namespace ov::intel_cpu::aarch64
