// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/expression.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "libxsmm.h"

namespace ov {
namespace intel_cpu {
// Note: The macro allows to automatically set appropriate environment variables for TPP/Libxsmm kernel compilation
// All TPP kernels must be compiled using this macro.
// * LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX enables more accurate exp approximation and exact division in TPP
// * LIBXSMM_GEMM_K_A_PF_DIST allows to tweak prefetching for GEMM kernels
#define COMPILE_TPP_KERNEL(...) \
    [&]() { \
        setenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX", "1", 1); \
        setenv("LIBXSMM_GEMM_K_A_PF_DIST", "4", 1); \
        auto res = reinterpret_cast<const uintptr_t>(__VA_ARGS__); \
        unsetenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX"); \
        unsetenv("LIBXSMM_GEMM_K_A_PF_DIST"); \
        return res; \
    }()
class DebugTppEmitter;
class TppEmitter : public jit_emitter {
    friend DebugTppEmitter;

public:
    TppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
               dnnl::impl::cpu::x64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);
    void emit_code(const std::vector<size_t> &in,
                   const std::vector<size_t> &out) const;
    static libxsmm_datatype ov_to_xsmm_dtype(ov::element::Type_t elemet_type);

protected:
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    static ov::snippets::VectorDims get_projected_subtensor(const snippets::lowered::PortDescriptorPtr& desc);

    /// Generate function pointer to the thin wrapper over the kernel that is called in runtime on every iteration
    virtual const uintptr_t get_execute_function_ptr() const = 0;
    /// Generate pointer to compiled kernel
    virtual const uintptr_t get_compiled_kernel_ptr() const = 0;

    std::vector<size_t> io_offsets{};
    std::vector<libxsmm_datatype> io_dtypes{};
    // Note: almost all emitters use fp32 for internal computations
    libxsmm_datatype exec_dtype {LIBXSMM_DATATYPE_F32};
    // aka leading dimensions
    std::vector<size_t> io_strides{};
    std::vector<snippets::lowered::PortDescriptorPtr> io_port_descriptors{};
    // compile flags has the same type for all eltwises, so we keep them in the base class
    libxsmm_bitfield m_compile_flags {0};
    int num_kernel_args = 0;
};

}   // namespace intel_cpu
}   // namespace ov
