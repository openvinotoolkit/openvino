// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_horizon_emitter : public jit_emitter {
public:
    jit_horizon_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                        ov::intel_cpu::riscv64::cpu_isa_t isa,
                        const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 1;
    }

protected:
    size_t aux_fp_gprs_count() const override {
        return 2;
    }

private:
    enum class OpType { max, sum };

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    template <ov::intel_cpu::riscv64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    OpType m_op_type = OpType::max;
};

}  // namespace ov::intel_cpu::riscv64
