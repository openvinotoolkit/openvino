// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_nop_emitter : public jit_emitter {
public:
    jit_nop_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                    ov::intel_cpu::riscv64::cpu_isa_t isa,
                    const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override {}
};

class jit_scalar_emitter : public jit_emitter {
public:
    jit_scalar_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                       ov::intel_cpu::riscv64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

protected:
    size_t aux_gprs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    int32_t value;
};

}  // namespace ov::intel_cpu::riscv64