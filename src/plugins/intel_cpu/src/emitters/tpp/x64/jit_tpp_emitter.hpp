// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/rt_info.hpp>

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/expression.hpp"

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "libxsmm.h"

namespace ov {
namespace intel_cpu {
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
    using jit_emitter::emit_code;
    void emit_impl(const std::vector<size_t>& in,
                   const std::vector<size_t>& out) const override;
    // Note: this method should be overriden in derived classes
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override {};
    static ov::snippets::VectorDims get_projected_subtensor(const snippets::lowered::PortDescriptorPtr& desc);

    /// Generate function pointer to the thin wrapper over the kernel that is called in runtime on every iteration
    virtual const uintptr_t get_execute_function_ptr() const = 0;
    /// Generate pointer to compiled kernel
    virtual const uintptr_t get_compiled_kernel_ptr() const = 0;

    std::vector<size_t> io_offsets{};
    std::vector<libxsmm_datatype> io_dtypes{};
    // Note: almost all emitters use fp32 for internal compulations
    const libxsmm_datatype dtype_comp {LIBXSMM_DATATYPE_F32};
    // aka leading dimensions
    std::vector<size_t> io_strides{};
    std::vector<snippets::lowered::PortDescriptorPtr> io_port_descriptors{};
    // compile flags has the same type for all eltwises, so we keep them in the base class
    libxsmm_bitfield m_compile_flags {0};
    int num_kernel_args = 0;
};

}   // namespace intel_cpu
}   // namespace ov
