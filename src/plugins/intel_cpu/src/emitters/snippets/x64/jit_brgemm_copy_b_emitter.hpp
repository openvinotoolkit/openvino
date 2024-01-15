// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>


namespace ov {
namespace intel_cpu {

class jit_brgemm_copy_b_emitter : public jit_emitter {
public:
    jit_brgemm_copy_b_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {return 1;}
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::i8}, {element::bf16}};
    }

private:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void init_brgemm_copy(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& kernel,
                          size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
                          bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const;
    void emit_kernel_call(const dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                          Xbyak::Reg64 src, Xbyak::Reg64 dst, Xbyak::Reg64 comp, size_t N, size_t K,
                          size_t offset_in, size_t offset_out, size_t offset_comp) const;

    static void execute(dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t* kernel,
                        const void* src, const void* dst, const void* comp, size_t N, size_t K);

    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> m_kernel;

    ov::element::Type m_brgemm_prc_in0, m_brgemm_prc_in1;
    size_t m_N, m_N_blk, m_N_tail;
    size_t m_K, m_K_blk, m_K_tail;
    size_t m_LDB;
    size_t m_brgemmVNNIFactor;
    bool m_with_comp = false;

    size_t m_in_offset = 0lu;
    size_t m_out_offset = 0lu;
    size_t m_comp_offset = 0lu;
};

}   // namespace intel_cpu
}   // namespace ov