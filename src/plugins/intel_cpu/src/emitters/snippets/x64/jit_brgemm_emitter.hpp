// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>


namespace ov {
namespace intel_cpu {

class jit_brgemm_emitter : public jit_emitter {
public:
    jit_brgemm_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return m_with_scratch ? 3 : 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

    static size_t get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
    static size_t get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    struct brgemmCtx {
        brgemmCtx() : M(0), N(0), K(0), LDA(0), LDB(0), LDC(0), dt_in0(dnnl_f32), dt_in1(dnnl_f32), is_with_amx(false), is_with_comp(false), beta(0) {}
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64] = {};
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };
    static void init_brgemm_kernel(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel, bool use_amx);

    void emit_brgemm_kernel_call(const dnnl::impl::cpu::x64::brgemm_kernel_t* brg_kernel, const brgemmCtx& ctx,
                                 Xbyak::Reg64 addr_A, Xbyak::Reg64 addr_B, Xbyak::Reg64 scratch, Xbyak::Reg64 addr_C,
                                 size_t in0_kernel_offset = 0, size_t in1_kernel_offset = 0,
                                 size_t in2_kernel_offset = 0, size_t out0_kernel_offset = 0) const;
    static void kernel_execute(const dnnl::impl::cpu::x64::brgemm_kernel_t *brg_kernel, const void *A, const void *B, void *C, void *scratch, int with_comp);

    brgemmCtx m_ctx;
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> m_kernel = nullptr;

    bool m_with_scratch = false;
    bool m_with_comp = false;

    size_t m_load_offset_a = 0lu;
    size_t m_load_offset_b = 0lu;
    size_t m_load_offset_scratch = 0lu;
    size_t m_store_offset_c = 0lu;

    std::vector<size_t> io_data_size {};
};

}   // namespace intel_cpu
}   // namespace ov
