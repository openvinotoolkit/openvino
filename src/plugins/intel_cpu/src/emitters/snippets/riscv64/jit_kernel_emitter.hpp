// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_kernel_emitter : public jit_emitter {
public:
    jit_kernel_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                       ov::intel_cpu::riscv64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    virtual void init_data_pointers(const std::vector<Xbyak_riscv::Reg>& arg_regs,
                                    const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
                                    const std::vector<Xbyak_riscv::Reg>& aux_gprs) const = 0;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    ov::intel_cpu::jit_snippets_compile_args jcp{};
    // gpr's used to store data pointers, track them to apply offsets in Kernel
    std::vector<size_t> data_ptr_regs_idx;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t num_unique_buffers = 0;

    std::shared_ptr<snippets::lowered::LinearIR> body;
};

class jit_kernel_static_emitter : public jit_kernel_emitter {
public:
    jit_kernel_static_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                              ov::intel_cpu::riscv64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override {
        return 2;
    }

private:
    void init_data_pointers(const std::vector<Xbyak_riscv::Reg>& arg_regs,
                            const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
                            const std::vector<Xbyak_riscv::Reg>& aux_gprs) const override;

    std::vector<size_t> master_shape;
    std::vector<std::vector<size_t>> data_offsets;
};

class jit_kernel_dynamic_emitter : public jit_kernel_emitter {
public:
    jit_kernel_dynamic_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                               ov::intel_cpu::riscv64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override {
        return 1;
    }

private:
    void init_data_pointers(const std::vector<Xbyak_riscv::Reg>& arg_regs,
                            const std::vector<Xbyak_riscv::Reg>& data_ptr_regs,
                            const std::vector<Xbyak_riscv::Reg>& aux_gprs) const override;
};

}  // namespace ov::intel_cpu::riscv64

