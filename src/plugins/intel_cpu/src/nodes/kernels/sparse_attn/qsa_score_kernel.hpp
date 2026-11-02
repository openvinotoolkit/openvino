// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

struct jit_qsa_score_compile_params {
    ov::element::Type row_prc;  // precision of the summary rows: f32, bf16 or f16
    size_t dim = 0;             // channels of a row, a multiple of the vector length
    size_t heads = 0;           // query heads of the index key head
};

struct jit_qsa_score_call_args {
    const float* q;    // [heads, dim] f32, the normalized and rotated query heads
    const void* rows;  // [n, dim] summary rows of row_prc
    float* scores;     // [n], scores[i] += sum over the heads h of max(0, q[h] . rows[i])
    float* stage;      // [kRows, dim] f32, scratch: the rows converted to f32 when row_prc is not f32
    size_t n;          // rows
};

// scores[i] += sum over the `heads` query heads of max(0, q[h] . rows[i]) for n summary rows of one index key head.
// Four rows at a time: the rows are converted to f32 once (they are shared by the heads), then per head the dot
// products of the four rows run on two independent accumulators each (the FMA latency would otherwise bound the loop),
// are reduced to scalars, passed through the ReLU and added to the rows' sums.
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
struct jit_qsa_score_kernel : public JitKernel<jit_qsa_score_compile_params, jit_qsa_score_call_args> {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_qsa_score_kernel)

    static constexpr size_t vec_size = dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float);
    static constexpr size_t kRows = 4;

    explicit jit_qsa_score_kernel(const jit_qsa_score_compile_params& jcp) : JitKernel(jit_name(), jcp, isa) {}

private:
    static_assert(isa == dnnl::impl::cpu::x64::avx2 || isa == dnnl::impl::cpu::x64::avx512_core,
                  "the scoring kernel is for AVX2 and above");
    using Vmm = std::conditional_t<isa == dnnl::impl::cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>;

    void generate() override;
    void emit_rows(size_t rows);
    void reduce_to_scalar(const Vmm& acc, const Vmm& tmp);
    void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, ov::element::Type src_prc, size_t offset);

    [[nodiscard]] Vmm acc(size_t row, size_t part) const {
        return Vmm(static_cast<int>(row * 2 + part));
    }
    [[nodiscard]] Xbyak::Xmm row_sum(size_t row) const {
        return Xbyak::Xmm(static_cast<int>(9 + row));
    }

    const Vmm vmm_q = Vmm(8);
    const Vmm vmm_tmp = Vmm(14);
    const Xbyak::Xmm xmm_zero = Xbyak::Xmm(13);
    const Xbyak::Xmm xmm_tmp = Xbyak::Xmm(15);

    const Xbyak::Reg64 reg_q = r8;
    const Xbyak::Reg64 reg_rows = r9;
    const Xbyak::Reg64 reg_scores = r10;
    const Xbyak::Reg64 reg_stage = r11;
    const Xbyak::Reg64 reg_n = r12;
    const Xbyak::Reg64 reg_rowbase = r13;
    const Xbyak::Reg64 reg_qh = r14;
    const Xbyak::Reg64 reg_heads = r15;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rax.getIdx()),
                                                   static_cast<size_t>(rdx.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {15};
};

}  // namespace ov::intel_cpu::kernel
