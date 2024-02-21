// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "jit_tpp_emitter.hpp"

namespace ov {
namespace intel_cpu {

class BrgemmTppEmitter : public TppEmitter {
public:
    BrgemmTppEmitter(dnnl::impl::cpu::x64::jit_generator* h,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

    static void execute_brgemm_kernel(libxsmm_gemmfunction brgemm_kernel, void *in0, void *in1, void *out0);

    const uintptr_t get_execute_function_ptr() const override { return reinterpret_cast<const uintptr_t>(execute_brgemm_kernel); }
    const uintptr_t get_compiled_kernel_ptr() const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    static void validate_subtensors(const VectorDims& in_0, const VectorDims& in_1, const VectorDims& out_0);
    libxsmm_gemm_shape m_shape;
    libxsmm_bitfield m_prefetching_flags {0};
};

}   // namespace intel_cpu
}   // namespace ov
