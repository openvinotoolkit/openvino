// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_tpp_emitter.hpp"
namespace ov {
namespace intel_cpu {

class BinaryEltwiseTppEmitter : public TppEmitter {
public:
    BinaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override { return 2; }
    static void execute_kernel(libxsmm_meltwfunction_binary eltwise_kernel, void *in0, void *in1, void *out0);
    const uintptr_t get_execute_function_ptr() const override { return reinterpret_cast<const uintptr_t>(execute_kernel); }
    const uintptr_t get_compiled_kernel_ptr() const override;
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    libxsmm_meltw_binary_shape m_shape;
    libxsmm_meltw_binary_type m_op_type;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
};

class UnaryEltwiseTppEmitter : public TppEmitter {
public:
    UnaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                            dnnl::impl::cpu::x64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override { return 1; }

    static void execute_kernel(libxsmm_meltwfunction_unary eltwise_kernel, void *in0, void *out0);
    const uintptr_t get_compiled_kernel_ptr() const override {
        return COMPILE_TPP_KERNEL(libxsmm_dispatch_meltw_unary(m_op_type,
                                                               m_shape,
                                                               m_compile_flags));
    }
    const uintptr_t get_execute_function_ptr() const override { return reinterpret_cast<const uintptr_t>(execute_kernel); }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    libxsmm_meltw_unary_shape m_shape;
    libxsmm_meltw_unary_type m_op_type;
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
};

class ReduceTppEmitter : public UnaryEltwiseTppEmitter {
public:
    ReduceTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);
};

class ReferenceUnaryEltwiseTppEmitter : public UnaryEltwiseTppEmitter {
public:
    // Note: can create template to suppport different executor signatures
    typedef std::function<float(float)> executor_function;
    ReferenceUnaryEltwiseTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                                    dnnl::impl::cpu::x64::cpu_isa_t isa,
                                    const ov::snippets::lowered::ExpressionPtr& expr,
                                    executor_function executor) :
                                    UnaryEltwiseTppEmitter(h, isa, expr), executor(std::move(executor)) {
    }
    static void execute_unary_eltw_kernel(ReferenceUnaryEltwiseTppEmitter* ref_emitter, void *in0, void *out0) {
        assert(ref_emitter);
        // Note: we can instantiate template with different precision combinations here, if we need to
        ref_emitter->evaluate_reference_impl(reinterpret_cast<float*>(in0), reinterpret_cast<float*>(out0));
    }
    const uintptr_t get_execute_function_ptr() const override {
        return reinterpret_cast<const uintptr_t>(execute_unary_eltw_kernel);
    }
    const uintptr_t get_compiled_kernel_ptr() const override {
        return reinterpret_cast<const uintptr_t>(this);
    }

private:
    executor_function executor{nullptr};

    template<class Tin, class Tout,
             typename std::enable_if<!std::is_same<Tin, Tout>::value || !std::is_same<Tin, float>::value, bool>::type = true>
    void evaluate_reference_impl(Tin* in0, Tout* out0) {
        for (int n = 0; n < m_shape.n; n++) {
                auto in0_row = in0;
                auto out0_row = out0;
                for (int m = 0; m < m_shape.m; m++)
                    out0_row[m] = static_cast<Tout>(executor(static_cast<float>(in0_row[m])));
                in0 += m_shape.ldi;
                out0 += m_shape.ldo;
        }
    }
    void evaluate_reference_impl(float* in0, float* out0) {
        for (int n = 0; n < m_shape.n; n++) {
                auto in0_row = in0;
                auto out0_row = out0;
                for (int m = 0; m < m_shape.m; m++)
                    out0_row[m] = executor(in0_row[m]);
                in0 += m_shape.ldi;
                out0 += m_shape.ldo;
        }
    }
};


}   // namespace intel_cpu
}   // namespace ov
