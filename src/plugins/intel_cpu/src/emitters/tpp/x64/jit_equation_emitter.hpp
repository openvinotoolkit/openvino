// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_tpp_emitter.hpp"

namespace ov::intel_cpu {

class EquationTppEmitter : public TppEmitter {
public:
    EquationTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                       dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override;
    static void execute_kernel(libxsmm_meqn_function equation_kernel, int argc, void** argv);
    const uintptr_t get_execute_function_ptr() const override {
        return reinterpret_cast<const uintptr_t>(execute_kernel);
    }
    const uintptr_t get_compiled_kernel_ptr() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    libxsmm_blasint m_equation_id;
    libxsmm_meqn_arg_shape m_out_shape;
    size_t m_num_inputs = 0;
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
};

}  // namespace ov::intel_cpu
