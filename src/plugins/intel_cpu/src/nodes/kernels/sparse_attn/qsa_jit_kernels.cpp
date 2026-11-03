// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_jit_kernels.hpp"

#include <cache/multi_cache.h>
#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "nodes/kernels/x64/jit_kernel_base.hpp"
#include "nodes/kernels/x64/rms_kernel.hpp"
#include "nodes/kernels/x64/rope_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

namespace {

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

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

#define GET_OFF(field) offsetof(jit_qsa_score_call_args, field)

template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::load(const Vmm& vmm_dst,
                                     const Xbyak::Reg64& reg_src,
                                     ov::element::Type src_prc,
                                     size_t offset) {
    constexpr auto load_num = static_cast<int>(vec_size);
    const auto seed = load_emitter_params(src_prc, ov::element::f32, load_num, false, "float_min").hash();
    if (!emitters[seed]) {
        emitters[seed] = std::make_unique<jit_load_emitter>(this,
                                                            isa,
                                                            src_prc,
                                                            ov::element::f32,
                                                            load_num,
                                                            ov::element::f32,
                                                            false,
                                                            "float_min");
    }
    emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), offset},
                              {static_cast<size_t>(vmm_dst.getIdx())},
                              pool_aux_vmm_idxs,
                              pool_aux_gpr_idxs);
}

// The sum of the lanes of `acc` into its lowest lane (`tmp` is clobbered).
template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::reduce_to_scalar(const Vmm& acc, const Vmm& tmp) {
    if constexpr (isa == avx512_core) {
        vextractf64x4(Ymm(tmp.getIdx()), Zmm(acc.getIdx()), 1);
        vaddps(Ymm(acc.getIdx()), Ymm(acc.getIdx()), Ymm(tmp.getIdx()));
    }
    vextractf128(Xmm(tmp.getIdx()), Ymm(acc.getIdx()), 1);
    vaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(tmp.getIdx()));
    vhaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(acc.getIdx()));
    vhaddps(Xmm(acc.getIdx()), Xmm(acc.getIdx()), Xmm(acc.getIdx()));
}

// The code for `rows` rows starting at reg_rows: scores[0, rows) += ...
template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::emit_rows(size_t rows) {
    const size_t dim = m_jcp.dim;
    const size_t chunks = dim / vec_size;
    const size_t parts = chunks > 1 ? 2 : 1;
    const bool convert = m_jcp.row_prc != ov::element::f32;
    const size_t row_bytes = dim * m_jcp.row_prc.size();

    // the rows as f32: in place, or converted once into the stage
    if (convert) {
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < chunks; c++) {
                load(vmm_q, reg_rows, m_jcp.row_prc, r * row_bytes + c * vec_size * m_jcp.row_prc.size());
                uni_vmovups(ptr[reg_stage + (r * dim + c * vec_size) * sizeof(float)], vmm_q);
            }
        }
        mov(reg_rowbase, reg_stage);
    } else {
        mov(reg_rowbase, reg_rows);
    }

    for (size_t r = 0; r < rows; r++) {
        uni_vpxor(row_sum(r), row_sum(r), row_sum(r));
    }
    mov(reg_qh, reg_q);
    mov(reg_heads, m_jcp.heads);

    Xbyak::Label head_loop;
    L(head_loop);
    {
        for (size_t r = 0; r < rows; r++) {
            for (size_t part = 0; part < parts; part++) {
                uni_vpxor(acc(r, part), acc(r, part), acc(r, part));
            }
        }
        for (size_t c = 0; c < chunks; c++) {
            uni_vmovups(vmm_q, ptr[reg_qh + c * vec_size * sizeof(float)]);
            for (size_t r = 0; r < rows; r++) {
                vfmadd231ps(acc(r, c % parts), vmm_q, ptr[reg_rowbase + (r * dim + c * vec_size) * sizeof(float)]);
            }
        }
        for (size_t r = 0; r < rows; r++) {
            if (parts == 2) {
                vaddps(acc(r, 0), acc(r, 0), acc(r, 1));
            }
            reduce_to_scalar(acc(r, 0), vmm_tmp);
            // max(0, q . row), summed over the heads
            vmaxss(Xmm(acc(r, 0).getIdx()), Xmm(acc(r, 0).getIdx()), xmm_zero);
            vaddss(row_sum(r), row_sum(r), Xmm(acc(r, 0).getIdx()));
        }
        add(reg_qh, dim * sizeof(float));
        dec(reg_heads);
        jnz(head_loop, T_NEAR);
    }

    for (size_t r = 0; r < rows; r++) {
        vmovss(xmm_tmp, ptr[reg_scores + r * sizeof(float)]);
        vaddss(xmm_tmp, xmm_tmp, row_sum(r));
        vmovss(ptr[reg_scores + r * sizeof(float)], xmm_tmp);
    }
}

template <cpu_isa_t isa>
void jit_qsa_score_kernel<isa>::generate() {
    this->preamble();
    mov(reg_q, ptr[abi_param1 + GET_OFF(q)]);
    mov(reg_rows, ptr[abi_param1 + GET_OFF(rows)]);
    mov(reg_scores, ptr[abi_param1 + GET_OFF(scores)]);
    mov(reg_stage, ptr[abi_param1 + GET_OFF(stage)]);
    mov(reg_n, ptr[abi_param1 + GET_OFF(n)]);
    uni_vpxor(xmm_zero, xmm_zero, xmm_zero);

    const size_t row_bytes = m_jcp.dim * m_jcp.row_prc.size();
    Xbyak::Label rows_loop;
    Xbyak::Label single_loop;
    Xbyak::Label done;

    L(rows_loop);
    cmp(reg_n, kRows);
    jl(single_loop, T_NEAR);
    emit_rows(kRows);
    add(reg_rows, kRows * row_bytes);
    add(reg_scores, kRows * sizeof(float));
    sub(reg_n, kRows);
    jmp(rows_loop, T_NEAR);

    L(single_loop);
    cmp(reg_n, 0);
    jle(done, T_NEAR);
    emit_rows(1);
    add(reg_rows, row_bytes);
    add(reg_scores, sizeof(float));
    dec(reg_n);
    jmp(single_loop, T_NEAR);

    L(done);
    this->postamble();
    for (const auto& emitter : emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

#undef GET_OFF

}  // namespace

}  // namespace ov::intel_cpu::kernel

namespace ov::intel_cpu::node::qsa {

namespace {

using dnnl::impl::cpu::x64::avx2;
using dnnl::impl::cpu::x64::avx512_core;
using dnnl::impl::cpu::x64::mayiuse;

std::shared_ptr<kernel::JitKernelBase> make_rms_kernel(const RmsRopeKey& key) {
    kernel::jit_rms_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.data_size = key.dim;
    jcp.scale_size = key.dim;
    jcp.eps = key.eps;
    std::shared_ptr<kernel::JitKernelBase> res;
    if (mayiuse(avx512_core)) {
        res = std::make_shared<kernel::jit_rms_kernel<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        res = std::make_shared<kernel::jit_rms_kernel<avx2>>(jcp);
    }
    if (res) {
        res->create_kernel();
    }
    return res;
}

std::shared_ptr<kernel::JitKernelBase> make_rotary_kernel(const RmsRopeKey& key) {
    kernel::jit_rotary_compile_params jcp;
    jcp.src_prc = ov::element::f32;
    jcp.dst_prc = ov::element::f32;
    jcp.rotary_ndims = key.rotary_dim;
    jcp.cos_sin_ndims = key.rotary_dim;
    jcp.mode = kernel::jit_rotary_compile_params::Mode::ROTATE_HALF;
    std::shared_ptr<kernel::JitKernelBase> res;
    if (mayiuse(avx512_core)) {
        res = std::make_shared<kernel::jit_rotary_kernel<avx512_core>>(jcp);
    } else if (mayiuse(avx2)) {
        res = std::make_shared<kernel::jit_rotary_kernel<avx2>>(jcp);
    }
    if (res) {
        res->create_kernel();
    }
    return res;
}

}  // namespace

std::shared_ptr<const JitRmsRope> JitRmsRope::get(const MultiCachePtr& cache, const RmsRopeKey& key) {
    const auto builder = [](const RmsRopeKey& k) -> std::shared_ptr<const JitRmsRope> {
        auto rms = make_rms_kernel(k);
        auto rotary = k.rotary_dim > 0 ? make_rotary_kernel(k) : nullptr;
        if (!rms || (k.rotary_dim > 0 && !rotary)) {
            return nullptr;
        }
        return std::make_shared<const JitRmsRope>(std::move(rms), std::move(rotary));
    };
    try {
        return cache->getOrCreate(key, builder).first;
    } catch (const ov::Exception&) {
        return nullptr;
    }
}

JitRmsRope::JitRmsRope(std::shared_ptr<kernel::JitKernelBase> rms, std::shared_ptr<kernel::JitKernelBase> rotary)
    : m_rms(std::move(rms)),
      m_rotary(std::move(rotary)) {}

void JitRmsRope::operator()(const float* src,
                            float* dst,
                            const float* weight,
                            const float* cos,
                            const float* sin) const {
    kernel::jit_rms_call_args rms_args{reinterpret_cast<const uint8_t*>(src), weight, reinterpret_cast<uint8_t*>(dst)};
    (*m_rms)(&rms_args);
    if (m_rotary) {
        kernel::jit_rotary_call_args rotary_args{dst, cos, sin, dst};
        (*m_rotary)(&rotary_args);
    }
}

bool JitQsaScore::supports(const ov::element::Type& row_prc, size_t dim) {
    if (row_prc != ov::element::f32 && row_prc != ov::element::bf16 && row_prc != ov::element::f16) {
        return false;
    }
    size_t vec_size = 0;
    if (mayiuse(avx512_core)) {
        vec_size = kernel::jit_qsa_score_kernel<avx512_core>::vec_size;
    } else if (mayiuse(avx2)) {
        vec_size = kernel::jit_qsa_score_kernel<avx2>::vec_size;
    }
    return vec_size != 0 && dim >= vec_size && dim % vec_size == 0;
}

std::shared_ptr<const JitQsaScore> JitQsaScore::get(const MultiCachePtr& cache, const QsaScoreKey& key) {
    if (!supports(key.row_prc, key.dim) || key.heads == 0) {
        return nullptr;
    }
    const auto builder = [](const QsaScoreKey& k) -> std::shared_ptr<const JitQsaScore> {
        kernel::jit_qsa_score_compile_params jcp;
        jcp.row_prc = k.row_prc;
        jcp.dim = k.dim;
        jcp.heads = k.heads;
        std::shared_ptr<kernel::JitKernelBase> res;
        if (mayiuse(avx512_core)) {
            res = std::make_shared<kernel::jit_qsa_score_kernel<avx512_core>>(jcp);
        } else {
            res = std::make_shared<kernel::jit_qsa_score_kernel<avx2>>(jcp);
        }
        res->create_kernel();
        return std::make_shared<const JitQsaScore>(std::move(res));
    };
    try {
        return cache->getOrCreate(key, builder).first;
    } catch (const ov::Exception&) {
        return nullptr;
    }
}

JitQsaScore::JitQsaScore(std::shared_ptr<kernel::JitKernelBase> kernel) : m_kernel(std::move(kernel)) {}

void JitQsaScore::operator()(const float* q, const void* rows, size_t n, float* scores, float* stage) const {
    kernel::jit_qsa_score_call_args args{q, rows, scores, stage, n};
    (*m_kernel)(&args);
}

}  // namespace ov::intel_cpu::node::qsa
