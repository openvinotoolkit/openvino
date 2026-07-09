// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::riscv64 {

class EmitABIRegSpills;
class jit_reg_spill_end_emitter;

class jit_reg_spill_begin_emitter : public jit_emitter {
    friend jit_reg_spill_end_emitter;

public:
    jit_reg_spill_begin_emitter(jit_generator_t* h,
                                cpu_isa_t isa,
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

    std::vector<snippets::Reg> m_regs_to_spill;
    std::shared_ptr<EmitABIRegSpills> m_abi_reg_spiller;
};

class jit_reg_spill_end_emitter : public jit_emitter {
public:
    jit_reg_spill_end_emitter(jit_generator_t* h,
                              cpu_isa_t isa,
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

    std::shared_ptr<EmitABIRegSpills> m_abi_reg_spiller;
};

}  // namespace ov::intel_cpu::riscv64
