// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

namespace ov::intel_cpu {

/* ================== jit_reg_spill_begin_emitters ====================== */
class EmitABIRegSpills;
class jit_reg_spill_end_emitter;
class jit_reg_spill_begin_emitter : public jit_emitter {
    friend jit_reg_spill_end_emitter;

public:
    jit_reg_spill_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                dnnl::impl::cpu::x64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;
    std::set<snippets::Reg> m_regs_to_spill;
    std::shared_ptr<EmitABIRegSpills> m_abi_reg_spiller;
};

/* ============================================================== */

/* ================== jit_reg_spill_end_emitter ====================== */

class jit_reg_spill_end_emitter : public jit_emitter {
public:
    jit_reg_spill_end_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                              dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    size_t aux_gprs_count() const override {
        return 0;
    }
    std::shared_ptr<EmitABIRegSpills> m_abi_reg_spiller;
};

/* ============================================================== */

}  // namespace ov::intel_cpu
