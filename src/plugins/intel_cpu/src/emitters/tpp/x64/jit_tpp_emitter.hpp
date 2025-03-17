// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/tpp/common/utils.hpp"
#include "libxsmm.h"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::intel_cpu {

class DebugTppEmitter;
class TppEmitter : public jit_binary_call_emitter {
    friend DebugTppEmitter;

public:
    TppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
               dnnl::impl::cpu::x64::cpu_isa_t isa,
               const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_code_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const;
    static ov::snippets::VectorDims get_projected_subtensor(const snippets::lowered::PortDescriptorPtr& desc);

    /// Generate function pointer to the thin wrapper over the kernel that is called in runtime on every iteration
    virtual const uintptr_t get_execute_function_ptr() const = 0;
    /// Generate pointer to compiled kernel
    virtual const uintptr_t get_compiled_kernel_ptr() const = 0;

    std::vector<size_t> io_offsets{};
    std::vector<libxsmm_datatype> io_dtypes{};
    // Note: almost all emitters use fp32 for internal computations
    libxsmm_datatype exec_dtype{LIBXSMM_DATATYPE_F32};
    // aka leading dimensions
    std::vector<size_t> io_strides{};
    std::vector<snippets::lowered::PortDescriptorPtr> io_port_descriptors{};
    // compile flags has the same type for all eltwises, so we keep them in the base class
    libxsmm_bitfield m_compile_flags{0};
    int num_kernel_args = 0;
};

}  // namespace ov::intel_cpu
