/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_brgemm_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_trans_m_k_f32_t : public jit_brgemm_trans_src_t,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_f32_t)

    jit_brgemm_trans_m_k_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float), transpose_size = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_K = r10;
    reg64_t reg_loop_M = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto load = [=](int i) {
        auto src_load = src_zmm(i);
        if (i >= nrows) {
            vpxord(src_load, src_load, src_load);
            return;
        }

        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        const bool partial_store = nrows < transpose_size;
        const auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        const auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i * 2;
            const int src_idx1 = src_idx0 + 1;

            const int next_src_idx0 = src_idx0 + 2;
            const int next_src_idx1 = src_idx1 + 2;
            const bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx1);
            const auto src0 = src_zmm(src_idx0);
            const auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            const int select_half = (i < 2) ? 0 : 2;
            const int src_idx0 = base_idx + i + select_half + 0;
            const int src_idx2 = src_idx0 + 2;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto tmp1 = tmp_zmm(src_idx2);
            const auto src0 = src_zmm(src_idx0);
            const auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i;
            const int src_idx4 = src_idx0 + 4;

            const auto tmp0 = tmp_zmm(src_idx0);
            const auto src0 = src_zmm(src_idx0);
            const auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [=]() {
        // swap 8
        const auto max_iters_phase_1 = std::min(ncolumns, 8);
        for (int i = 0; i < max_iters_phase_1; i++) {
            const auto tmp = tmp_zmm(i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        const auto max_iters_phase_2 = std::min(ncolumns - 8, 8);
        for (int i = 0; i < max_iters_phase_2; i++) {
            const auto tmp = tmp_zmm(8 + i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_trans_m_k_f32_t::generate() {
    preamble();
    assert(conf_->ic_block % transpose_size == 0);
    const int os_block = conf_->os_block;
    const int last_os_block_tail = conf_->K_tail % transpose_size;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = conf_->ic * typesize;
    tr_src_stride = conf_->LDA * typesize;
    const dim_t m_src_shift = transpose_size * typesize;
    const dim_t m_tr_src_shift = tr_src_stride * transpose_size;

    const dim_t batch_src_shift = src_stride * os_block;
    const dim_t batch_tr_src_shift = tr_src_stride * conf_->M;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_M = [=](bool is_os_tail) {
        const auto nrows = is_os_tail ? last_os_block_tail : transpose_size;
        mov(reg_loop_M, ptr[param1 + GET_OFF(current_M)]);
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label M_loop, M_tail_or_done, M_done;
        if (ic_tail > 0) {
            cmp(reg_loop_M, transpose_size);
            jl(M_tail_or_done, T_NEAR);
        }

        L(M_loop);
        transpose_16x16(nrows, transpose_size);
        if (conf_->ic_block > transpose_size) {
            add(reg_src, m_src_shift);
            add(reg_tr_src, m_tr_src_shift);
            sub(reg_loop_M, transpose_size);
            cmp(reg_loop_M, transpose_size);
            jge(M_loop, T_NEAR);
        } else {
            jmp(M_done, T_NEAR);
        }

        L(M_tail_or_done);
        if (ic_tail > 0) {
            cmp(reg_loop_M, 0);
            jle(M_done, T_NEAR);

            transpose_16x16(nrows, ic_tail);
        }
        L(M_done);
    };

    auto compute_batch = [=](bool is_os_tail) {
        Label batch_loop;
        L(batch_loop);

        compute_M(is_os_tail);
        add(reg_src_base, batch_src_shift);
        add(reg_tr_src_base, batch_tr_src_shift);

        sub(reg_loop_batch, 1);
        jnz(batch_loop, T_NEAR);
    };

    Label K_tail;
    if (last_os_block_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    compute_batch(false);

    if (last_os_block_tail > 0) {
        Label K_done;
        jmp(K_done, T_NEAR);

        L(K_tail);
        compute_batch(true);
        L(K_done);
    }

    postamble();
}

struct jit_brgemm_trans_m_k_bf16_t : public jit_brgemm_trans_src_t,
                                     public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_m_k_bf16_t)
    jit_brgemm_trans_m_k_bf16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_src_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum {
        typesize = sizeof(int16_t),
        transpose_size = 16,
    };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kFFFF = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t k55 = k5;
    opmask_t kCC = k6;
    opmask_t k33 = k7;
    opmask_t kTail = k1;

    reg32_t regw_tmp = r15d;

    reg64_t reg_k_src = r14;
    reg64_t reg_k_tr_src = r13;

    reg64_t reg_m_src = r12;
    reg64_t reg_m_tr_src = r11;

    reg64_t reg_batch_src = r10;
    reg64_t reg_batch_tr_src = r9;

    reg64_t reg_loop_batch = r8;
    reg64_t reg_loop_K = rax;
    reg64_t reg_loop_M = rbx;

    reg64_t reg_tr_src_tmp = abi_not_param1; // lnx -> rcx
    reg64_t imm_addr64 = rdx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    void transpose(
            reg64_t dst, reg64_t src, int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_m_k_bf16_t::transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) { return Zmm(i); };

    auto src_ymm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, dst);

        auto k = kTail;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    const int ic_block = ncolumns;
    kmovd(kFFFF, ic_block < transpose_size ? (1 << ic_block) - 1 : 0xffff);

    for (int i = 0; i < nrows / 2; i++) {
        auto zmm_src0 = src_zmm(2 * i);
        auto zmm_src1 = src_zmm(2 * i + 1);
        auto src1 = src_ymm(2 * i + 1);
        vmovdqu16(zmm_src0 | kFFFF | T_z,
                EVEX_compress_addr(src, 2 * i * src_stride));
        vmovdqu16(zmm_src1 | kFFFF | T_z,
                EVEX_compress_addr(src, (2 * i + 1) * src_stride));
        vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    // for odd numbers we need to mix row with zeroes
    if (nrows % 2) {
        int i = nrows / 2;
        auto zmm_src0 = src_zmm(2 * i);
        vmovdqu16(zmm_src0 | kFFFF | T_z,
                EVEX_compress_addr(src, 2 * i * src_stride));
        vpermw(zmm_src0, vidx5, zmm_src0);
    }

    for (int i = rnd_up(nrows, 2); i < 16; i += 2) {
        vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    }

    // swap 1
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(4 * i);
        auto zmm1 = src_zmm(4 * i + 2);
        auto tmp0 = src_zmm(4 * i + 1);
        auto tmp1 = src_zmm(4 * i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kAAAA, vidx3, zmm1);
        vpermps(tmp1 | k5555, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 2; i++) {
        auto zmm0 = src_zmm(base_idx + 2 * i + 1);
        auto zmm1 = src_zmm(base_idx + 2 * i + 5);

        auto tmp0 = src_zmm(base_idx + 2 * i);
        auto tmp1 = src_zmm(base_idx + 2 * i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kAA, vidx2, zmm1);
        vpermpd(tmp1 | k55, vidx2, zmm0);
    }

    // swap 3
    for (int i = 0; i < 4; i++) {
        auto zmm0 = src_zmm(2 * i);
        auto zmm1 = src_zmm(2 * i + 8);

        auto tmp0 = src_zmm(2 * i + 1);
        auto tmp1 = src_zmm(2 * i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kCC, vidx1, zmm1);
        vpermpd(tmp1 | k33, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 8; i++)
        vextracti64x4(src_ymm(2 * i), src_zmm(2 * i + 1), 1);

    auto get_vec_idx = [=](int ic_idx) {
        assert(ic_idx < 16 && ic_idx >= 0);
        switch (ic_idx) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
            case 4: return 9;
            case 5: return 8;
            case 6: return 11;
            case 7: return 10;
            case 8: return 5;
            case 9: return 4;
            case 10: return 7;
            case 11: return 6;
            case 12: return 13;
            case 13: return 12;
            case 14: return 15;
            default: return 14;
        }
    };

    int store_tail = rnd_up(nrows, 2);
    kmovw(kTail, (1 << store_tail / 2) - 1);

    for (int ic = 0; ic < ic_block; ic++)
        store(src_zmm(get_vec_idx(ic)), ic);
}

void jit_brgemm_trans_m_k_bf16_t::generate() {
    preamble();

    alignas(64) static constexpr const int64_t idx1[8]
            = {2, 3, 0, 1, 6, 7, 4, 5};
    alignas(64) static constexpr const int64_t idx2[8]
            = {1, 0, 3, 2, 5, 4, 7, 6};
    alignas(64) static constexpr const int32_t idx3[16]
            = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
    alignas(64) static constexpr const int32_t idx4[16]
            = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
    alignas(64) static constexpr const uint16_t idx5[32]
            = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1, 17,
                    3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

    constexpr int amx_bf16_granularity = 2;
    const bool last_row_padded = conf_->isa == avx512_core_bf16_amx_bf16
            && conf_->os % amx_bf16_granularity != 0;
    const int eff_K_tail = conf_->K_tail - (last_row_padded ? 1 : 0);

    const int os_block = conf_->os_block;
    const int last_os_block_tail = eff_K_tail % transpose_size;
    const int ic_tail = conf_->M_tail % transpose_size;
    src_stride = conf_->ic * typesize;
    tr_src_stride = conf_->LDA * typesize;

    const dim_t batch_src_shift = src_stride * os_block;
    const dim_t batch_tr_src_shift = tr_src_stride * conf_->M;

    const dim_t M_src_shift = transpose_size * typesize;
    const dim_t M_tr_src_shift = transpose_size * conf_->LDA * typesize;

    const dim_t K_src_shift = transpose_size * conf_->ic * typesize;
    const dim_t K_tr_src_shift = transpose_size * typesize;

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff);
    kmovw(k5555, 0x5555);
    kmovw(kAAAA, 0xaaaa);
    kmovw(kAA, 0xaa);
    kmovw(k55, 0x55);
    kmovw(kCC, 0xcc);
    kmovw(k33, 0x33);

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [=](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, idx1);
    vmovdqa64(vidx2, idx2);
    vmovdqa32(vidx3, idx3);
    vmovdqa32(vidx4, idx4);
    vmovdqa32(vidx5, (const int32_t *)idx5);

    auto compute_m_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base,
                                  bool is_os_tail) {
        mov(reg_loop_M, ptr[param1 + GET_OFF(current_M)]);
        mov(reg_m_src, reg_base);
        mov(reg_m_tr_src, reg_tr_base);

        Label M_loop_tail, M_loop;
        if (ic_tail > 0) {
            cmp(reg_loop_M, transpose_size);
            jl(M_loop_tail, T_NEAR);
        }
        L(M_loop);
        {
            transpose(reg_m_tr_src, reg_m_src,
                    is_os_tail ? last_os_block_tail : transpose_size,
                    transpose_size);
            add(reg_m_src, M_src_shift);
            add(reg_m_tr_src, M_tr_src_shift);
        }
        sub(reg_loop_M, transpose_size);
        cmp(reg_loop_M, transpose_size);
        jge(M_loop, T_NEAR);

        if (ic_tail > 0) {
            Label M_loop_done;
            L(M_loop_tail);
            cmp(reg_loop_M, 0);
            jle(M_loop_done, T_NEAR);

            transpose(reg_m_tr_src, reg_m_src,
                    is_os_tail ? last_os_block_tail : transpose_size, ic_tail);
            L(M_loop_done);
        }
    };

    auto compute_k_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base) {
        mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);
        mov(reg_k_src, reg_base);
        mov(reg_k_tr_src, reg_tr_base);

        Label K_tail, K_loop, K_done;
        if (last_os_block_tail > 0) {
            cmp(reg_loop_K, transpose_size);
            jl(K_tail, T_NEAR);
        }
        L(K_loop);
        {
            compute_m_loop(reg_k_src, reg_k_tr_src, false);
            add(reg_k_src, K_src_shift);
            add(reg_k_tr_src, K_tr_src_shift);
        }
        sub(reg_loop_K, transpose_size);
        cmp(reg_loop_K, transpose_size);
        jge(K_loop, T_NEAR);

        cmp(reg_loop_K, 0);
        je(K_done, T_NEAR);

        if (last_os_block_tail > 0) {
            L(K_tail);
            compute_m_loop(reg_k_src, reg_k_tr_src, true);
        }
        L(K_done);
    };

    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_batch_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_batch_tr_src, ptr[param1 + GET_OFF(tr_src)]);

    Label batch_loop;
    L(batch_loop);
    {
        compute_k_loop(reg_batch_src, reg_batch_tr_src);

        add(reg_batch_src, batch_src_shift);
        add(reg_batch_tr_src, batch_tr_src_shift);
    }
    sub(reg_loop_batch, 1);
    jnz(batch_loop, T_NEAR);

    postamble();
}

void jit_brgemm_copy_to_coarse_t::copy_row_blks(int num_row_blks) {
    int rnd_row_blks = div_up(num_row_blks, row_loop_unroll);

    for (int row_b = 0; row_b < rnd_row_blks; ++row_b) {
        const int row_start = 0;
        const int row_end = nstl::min(static_cast<int>(row_loop_unroll),
                num_row_blks - row_b * static_cast<int>(row_loop_unroll));

        for (int row = row_start; row < row_end; ++row) {
            const int row_idx = row_b * row_loop_unroll + row;
            const auto offset = addr_offset(row_idx);

            const auto zmm = get_zmm_copy(row);
            const auto addr = EVEX_compress_addr(reg_data, offset);
            const auto addr_tr = EVEX_compress_addr(reg_tr_data, offset);

            vmovdqu8(zmm, addr);
            vmovdqu8(addr_tr, zmm);
        }
    }
}

void jit_brgemm_copy_to_coarse_t::copy_row_tail(
        bool is_last_iteration, int row_offset) {
    // Masks for row tail load and store are already set up
    const auto load_mask = is_last_iteration ? reg_m_last_row_tail_load
                                             : reg_m_full_row_tail_load;
    const auto store_mask = is_last_iteration ? reg_m_last_row_tail_store
                                              : reg_m_full_row_tail_store;

    const auto zmm_data = zmm_row_tail | load_mask | T_z;
    const auto zmm_tr_data = zmm_row_tail | store_mask;

    const auto offset = addr_offset(row_offset);
    const auto addr = EVEX_compress_addr(reg_data, offset);
    const auto addr_tr = EVEX_compress_addr(reg_tr_data, offset);

    vmovdqu8(zmm_data, addr);
    vmovdqu8(addr_tr, zmm_tr_data);
}

void jit_brgemm_copy_to_coarse_t::zero_out_rows() {
    const int row_blk = row_size_ % tr_row_size_;
    const int rnd_up_row_blk = utils::rnd_up(row_blk, row_step_);

    int zero_row_blks = tr_row_size_ - rnd_up_row_blk;
    if (zero_row_blks == 0) return;

    const auto zmm_step = row_step_, ymm_step = row_step_ / 2,
               xmm_step = row_step_ / 4;
    assert(zero_row_blks % xmm_step == 0);
    MAYBE_UNUSED(xmm_step);

    int zmm_iters = zero_row_blks / zmm_step;
    zero_row_blks %= zmm_step;
    int ymm_iters = zero_row_blks / ymm_step;
    zero_row_blks %= ymm_step;
    int xmm_iters = zero_row_blks / xmm_step;

    auto offset = addr_offset(rnd_up_row_blk / row_step_);

    for (int row = 0; row < zmm_iters; ++row) {
        const auto addr_tr = EVEX_compress_addr(reg_tr_data, offset);
        vmovdqu8(addr_tr, zmm_zero);
        offset += (zmm_step * typesize_);
    }

    const auto ymm_zero = Xbyak::Ymm(zmm_zero.getIdx());
    const auto xmm_zero = Xbyak::Xmm(zmm_zero.getIdx());

    assert(xmm_iters <= 1 && ymm_iters <= 1);
    if (ymm_iters > 0) {
        const auto addr_tr = EVEX_compress_addr(reg_tr_data, offset);
        vmovdqu8(addr_tr, ymm_zero);
        offset += (ymm_step * typesize_);
    }

    if (xmm_iters > 0) {
        const auto addr_tr = EVEX_compress_addr(reg_tr_data, offset);
        vmovdqu8(addr_tr, xmm_zero);
    }
}

void jit_brgemm_copy_to_coarse_t::copy_row_loop() {
    Xbyak::Label label_row_tail, label_row_exit;

    // Note: copying is done in chunks of size row_step_
    const auto copy_row = [&](bool is_last_iteration) {
        const int row_blk
                = is_last_iteration ? (row_size_ % tr_row_size_) : tr_row_size_;
        const int row_iters = row_blk / row_step_;
        const int row_iters_tail = row_blk % row_step_;

        copy_row_blks(row_iters);
        if (row_iters_tail != 0)
            copy_row_tail(is_last_iteration, /* row_offset = */ row_iters);

        // For the last iteration, zero-out rows if needed
        if (is_last_iteration) zero_out_rows();
    };

    const bool only_row_tail = row_size_ < tr_row_size_;

    if (!only_row_tail) {
        cmp(reg_last_row_blk, 0);
        jne(label_row_tail, T_NEAR);

        copy_row(/* is_last_iteration = */ false);
        jmp(label_row_exit, T_NEAR);
    }

    L(label_row_tail);
    copy_row(/* is_last_iteration = */ true);

    L(label_row_exit);
}

void jit_brgemm_copy_to_coarse_t::copy_os_loop() {

    Label loop_os;
    L(loop_os);

    copy_row_loop();
    add(reg_data, data_stride_);
    add(reg_tr_data, tr_data_stride_);

    dec(reg_os_work);
    jnz(loop_os, T_NEAR);
}

void jit_brgemm_copy_to_coarse_t::set_last_row_tail_masks() {
    const int row_tail = (row_size_ % tr_row_size_) % row_step_;
    assert(row_tail > 0 && "kernel is meant to be used with tail processing");

    // Set load mask
    const size_t tail_mask_load
            = (static_cast<size_t>(1) << (typesize_ * row_tail)) - 1;
    mov(reg_tail_mask, tail_mask_load);
    kmovq(reg_m_last_row_tail_load, reg_tail_mask);

    // Caution: Since size of ZMM equals 64 bytes therefore we need
    // different masks to store tails with smaller row_block_size_
    constexpr auto full_mask = size_t {0xffffffffffffffff};
    constexpr auto half_mask = size_t {0x00000000ffffffff};
    constexpr auto quad_mask = size_t {0x000000000000ffff};

    const auto num_bytes = [](size_t mask) -> int {
        // Given by 1 + position of leftmost 1 bit
        return 1 + math::ilog2q(mask);
    };

    const int row_tail_store_size
            = utils::rnd_up(row_tail, row_block_size_) * typesize_;
    if (row_tail_store_size >= num_bytes(full_mask))
        mov(reg_tail_mask, full_mask);
    else if (row_tail_store_size >= num_bytes(half_mask))
        mov(reg_tail_mask, half_mask);
    else {
        assert(row_tail_store_size == num_bytes(quad_mask));
        mov(reg_tail_mask, quad_mask);
    }
    kmovq(reg_m_last_row_tail_store, reg_tail_mask);
}

void jit_brgemm_copy_to_coarse_t::set_full_row_tail_masks() {
    const auto full_row_tail = tr_row_size_ % row_step_;
    assert(row_step_ == 2 * full_row_tail || row_step_ == 4 * full_row_tail);

    const auto tail_mask = row_step_ == 2 * full_row_tail
            ? size_t {0x00000000ffffffff}
            : size_t {0x000000000000ffff};

    mov(reg_tail_mask, tail_mask);
    kmovq(reg_m_full_row_tail_store, reg_tail_mask);
    kmovq(reg_m_full_row_tail_load, reg_tail_mask);
}

void jit_brgemm_copy_to_coarse_t::generate() {
    preamble();

    // set up masks for tail processing
    set_last_row_tail_masks();
    const bool has_full_row_tail_ = tr_row_size_ % row_step_ != 0;
    if (has_full_row_tail_) set_full_row_tail_masks();

    // init zero vreg (zmm_zero) if it is needed
    const int last_row_size
            = utils::rnd_up(row_size_ % tr_row_size_, row_step_);
    const bool zero_iters_needed
            = last_row_size > 0 && last_row_size < tr_row_size_;
    if (zero_iters_needed) vpxord(zmm_zero, zmm_zero, zmm_zero);

    // load arguments to the jit kernel
    mov(reg_data, ptr[param1 + GET_OFF(data)]);
    mov(reg_tr_data, ptr[param1 + GET_OFF(tr_data)]);
    mov(reg_os_work, ptr[param1 + GET_OFF(os_work)]);
    mov(reg_last_row_blk, ptr[param1 + GET_OFF(last_row_blk)]);

    // enter the `main` loop
    copy_os_loop();

    postamble();
}

struct jit_trans_to_vnni_t : public jit_brgemm_trans_to_vnni_t,
                             public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_to_vnni_t)
    jit_trans_to_vnni_t(const jit_brgemm_primitive_conf_t *conf,
            jit_brgemm_trans_to_vnni_t::matrix_to_transform_t
                    matrix_to_transform)
        : jit_brgemm_trans_to_vnni_t(conf, matrix_to_transform) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum {
        typesize_data = sizeof(int16_t),
        typesize_acc = sizeof(float),
        transpose_size = 16,
    };

    int last_row_block_tail = 0, col_tail = 0;
    dim_t src_stride = 0, tr_src_stride = 0;
    dim_t src_col_shift = 0, tr_src_col_shift = 0;
    dim_t src_row_shift = 0, tr_src_row_shift = 0;
    dim_t src_batch_shift = 0, tr_src_batch_shift = 0;

    opmask_t kFFFF = k1;
    opmask_t mask_tail = k2;

    zmm vidx1 = zmm31;

    reg32_t regw_tmp = r15d;

    reg64_t reg_batch_src = r14;
    reg64_t reg_batch_tr_src = r13;

    reg64_t reg_row_src = r12;
    reg64_t reg_row_tr_src = r11;

    reg64_t reg_col_src = r10;
    reg64_t reg_col_tr_src = r9;

    reg64_t reg_loop_batch = r8;
    reg64_t reg_loop_row = rax;
    reg64_t reg_loop_col = rbx;

    reg64_t imm_addr64 = abi_not_param1; // lnx -> rcx

    void maybe_zero_pad_col(reg64_t dst);
    void transpose(reg64_t dst, reg64_t src, int nrows,
            int ncolumns = transpose_size, bool pad_by_zeroes = false);
    void generate() override;
};

void jit_trans_to_vnni_t::maybe_zero_pad_col(reg64_t dst) {
    auto zmm_zero = Xbyak::Zmm(0);
    vpxord(zmm_zero, zmm_zero, zmm_zero);
    const int oc_utilized = rnd_up(conf_->oc % conf_->oc_block, transpose_size);
    const int iters = (conf_->oc_block - oc_utilized) / transpose_size;
    for (int n = 0; n < iters; ++n) {
        for (int i = 0; i < transpose_size; i += 2) {
            auto addr = EVEX_compress_addr(dst, i * tr_src_stride);
            vmovups(addr, zmm_zero);
        }
        add(reg_col_tr_src, tr_src_col_shift);
    }
}

void jit_trans_to_vnni_t::transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns, bool pad_by_zeroes) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) { return Zmm(i); };

    auto src_ymm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto store = [=](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * tr_src_stride);
        vmovups(addr, r);
    };
    auto mask = ncolumns == transpose_size ? kFFFF : mask_tail;

    int i = 0;
    for (; i < nrows / 2; i++) {
        auto src1 = src_ymm(2 * i + 1);
        auto zmm_src0 = src_zmm(2 * i);
        auto zmm_src1 = src_zmm(2 * i + 1);
        if (matrix_to_transform_ == matrix_B) {
            vmovdqu16(zmm_src0 | mask | T_z,
                    EVEX_compress_addr(src, 2 * i * src_stride));
            vmovdqu16(zmm_src1 | mask | T_z,
                    EVEX_compress_addr(src, (2 * i + 1) * src_stride));
            vinsertf64x4(zmm_src0, zmm_src0, src1, 1);
        } else {
            vmovups(zmm_src0 | mask | T_z,
                    EVEX_compress_addr(src, 2 * i * src_stride));
            vmovups(zmm_src1 | mask | T_z,
                    EVEX_compress_addr(src, (2 * i + 1) * src_stride));
            vcvtne2ps2bf16(zmm_src0, zmm_src1, zmm_src0);
        }
        vpermw(zmm_src0, vidx1, zmm_src0);
        store(zmm_src0, 2 * i);
    }

    if (nrows % 2) {
        auto zmm_src0 = src_zmm(2 * i);
        if (matrix_to_transform_ == matrix_B) {
            vmovdqu16(zmm_src0 | mask | T_z,
                    EVEX_compress_addr(src, 2 * i * src_stride));
        } else {
            auto zmm_zero = src_zmm(2 * i + 1);
            vmovups(zmm_src0 | mask | T_z,
                    EVEX_compress_addr(src, 2 * i * src_stride));
            vpxord(zmm_zero, zmm_zero, zmm_zero);
            vcvtne2ps2bf16(zmm_src0, zmm_zero, zmm_src0);
        }
        vpermw(zmm_src0, vidx1, zmm_src0);
        store(zmm_src0, 2 * i);
        i++;
    }

    if (pad_by_zeroes && i < transpose_size / 2) {
        auto zmm_zero = src_zmm(2 * i);
        vpxord(zmm_zero, zmm_zero, zmm_zero);
        for (; i < transpose_size / 2; i++)
            store(zmm_zero, 2 * i);
    }
}

void jit_trans_to_vnni_t::generate() {
    preamble();

    alignas(64) static constexpr const int16_t idx1[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    if (matrix_to_transform_ == matrix_B) {
        int row_block = conf_->os_block;

        constexpr int amx_bf16_granularity = 2;
        const bool last_row_padded = conf_->isa == avx512_core_bf16_amx_bf16
                && conf_->os % amx_bf16_granularity != 0;
        const int eff_K_tail = conf_->K_tail - (last_row_padded ? 1 : 0);

        last_row_block_tail = eff_K_tail % transpose_size;
        col_tail = conf_->oc % transpose_size;
        src_stride = conf_->oc * typesize_data;
        tr_src_stride = conf_->LDB * typesize_data;

        src_batch_shift = src_stride * row_block;
        tr_src_batch_shift = tr_src_stride * rnd_up(conf_->K, 2);

        src_col_shift = transpose_size * typesize_data;
        tr_src_col_shift = 2 * transpose_size * typesize_data;

        src_row_shift = transpose_size * conf_->oc * typesize_data;
        tr_src_row_shift = transpose_size * conf_->LDB * typesize_data;

    } else { // matrix_to_transform_ == matrix_C
        int row_block = conf_->ic_block;
        last_row_block_tail = conf_->M_tail % transpose_size;
        assert(row_block == transpose_size);
        col_tail = conf_->oc % transpose_size;
        src_stride = conf_->LDC * typesize_acc;
        tr_src_stride = conf_->LDD * typesize_data;

        src_batch_shift = src_stride * row_block;
        tr_src_batch_shift = tr_src_stride * rnd_up(conf_->M, 2);

        src_col_shift = transpose_size * typesize_acc;
        tr_src_col_shift = 2 * transpose_size * typesize_data;
    }

    //    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    //    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    //    mov(reg_loop_row, ptr[param1 + GET_OFF(current_row_size)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };
    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff); // 1111111111111111
    kmovd(mask_tail, (1 << col_tail) - 1);

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(vidx1, (const int64_t *)idx1);

    auto compute_col_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base,
                                    bool is_row_tail) {
        const bool pad_by_zeroes = matrix_to_transform_ == matrix_C;
        int nrows = is_row_tail ? last_row_block_tail : transpose_size;

        mov(reg_col_src, reg_base);
        mov(reg_col_tr_src, reg_tr_base);
        mov(reg_loop_col, ptr[param1 + GET_OFF(current_col_size)]);

        Label col_loop, col_loop_tail;
        cmp(reg_loop_col, transpose_size);
        jl(col_loop_tail, T_NEAR);

        L(col_loop);
        {
            transpose(reg_col_tr_src, reg_col_src, nrows, transpose_size,
                    pad_by_zeroes);
            add(reg_col_src, src_col_shift);
            add(reg_col_tr_src, tr_src_col_shift);
        }
        sub(reg_loop_col, transpose_size);
        cmp(reg_loop_col, transpose_size);
        jge(col_loop, T_NEAR);

        L(col_loop_tail);
        if (col_tail > 0) {
            Label col_loop_done;
            cmp(reg_loop_col, 0);
            jle(col_loop_done, T_NEAR);
            transpose(reg_col_tr_src, reg_col_src, nrows, col_tail,
                    pad_by_zeroes);
            L(col_loop_done);
        }
        const int oc_block_tail = conf_->oc % conf_->oc_block;
        const bool full_oc_block_utilized = oc_block_tail == 0
                || rnd_up(oc_block_tail, transpose_size) == conf_->oc_block;
        const bool col_pad_required = pad_by_zeroes && !full_oc_block_utilized;

        if (col_pad_required) {
            Label col_pad_done;
            mov(reg_loop_col, ptr[param1 + GET_OFF(current_col_size)]);
            cmp(reg_loop_col, conf_->oc_block);
            je(col_pad_done, T_NEAR);
            if (col_tail > 0) add(reg_col_tr_src, tr_src_col_shift);
            maybe_zero_pad_col(reg_col_tr_src);
            L(col_pad_done);
        }
    };

    auto compute_row_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base) {
        mov(reg_row_src, reg_base);
        mov(reg_row_tr_src, reg_tr_base);
        mov(reg_loop_row, ptr[param1 + GET_OFF(current_row_size)]);

        Label row_tail, row_loop, row_done;
        if (last_row_block_tail > 0) {
            cmp(reg_loop_row, transpose_size);
            jl(row_tail, T_NEAR);
        }
        L(row_loop);
        {
            compute_col_loop(reg_row_src, reg_row_tr_src, false);

            add(reg_row_src, src_row_shift);
            add(reg_row_tr_src, tr_src_row_shift);
        }
        sub(reg_loop_row, transpose_size);
        cmp(reg_loop_row, transpose_size);
        jge(row_loop, T_NEAR);

        cmp(reg_loop_row, 0);
        je(row_done, T_NEAR);

        if (last_row_block_tail > 0) {
            L(row_tail);
            compute_col_loop(reg_row_src, reg_row_tr_src, true);
        }
        L(row_done);
    };

    mov(reg_batch_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_batch_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);

    Label batch_loop;
    L(batch_loop);
    {
        compute_row_loop(reg_batch_src, reg_batch_tr_src);

        add(reg_batch_src, src_batch_shift);
        add(reg_batch_tr_src, tr_src_batch_shift);
    }
    sub(reg_loop_batch, 1);
    jnz(batch_loop, T_NEAR);

    postamble();
}

struct jit_copy_f32_t : public jit_brgemm_trans_to_vnni_t,
                        public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_copy_f32_t)
    jit_copy_f32_t(const jit_brgemm_primitive_conf_t *conf,
            jit_brgemm_trans_to_vnni_t::matrix_to_transform_t
                    matrix_to_transform)
        : jit_brgemm_trans_to_vnni_t(conf, matrix_to_transform) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum {
        typesize_data = sizeof(float),
        column_step = 16,
        num_regs = 32,
    };

    dim_t src_stride = 0, tr_src_stride = 0;
    dim_t src_batch_shift = 0, tr_src_batch_shift = 0;
    dim_t col_shift = column_step * typesize_data;

    opmask_t mask_tail = k2;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_batch = r10;
    reg64_t reg_loop_row = r11;
    reg64_t reg_loop_col = r12;
    reg32_t regw_tmp = r14d;
    reg64_t reg_long_offt = r15;

    void copy_block(int nrows, int ncolumns);
    void generate() override;
};

void jit_copy_f32_t::copy_block(int nrows, int ncolumns) {

    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    const int nc_tail = ncolumns % column_step;
    if (nc_tail > 0) kmovd(mask_tail, (1 << nc_tail) - 1);

    auto get_zmm = [=](int i) { return Zmm(i % num_regs); };

    auto load = [=](int r, int cb) {
        auto src_reg = get_zmm(r * cb);
        const bool is_tail
                = nc_tail > 0 && ncolumns - cb * column_step < column_step;
        auto src_load = is_tail ? src_reg | mask_tail | T_z : src_reg;
        const dim_t offset = r * src_stride + cb * col_shift;
        auto addr = EVEX_compress_addr_safe(reg_src, offset, reg_long_offt);
        vmovups(src_load, addr);
    };

    auto store = [=](int r, int cb) {
        auto reg = get_zmm(r * cb);
        const dim_t offset = r * tr_src_stride + cb * col_shift;
        auto addr = EVEX_compress_addr_safe(reg_tr_src, offset, reg_long_offt);
        vmovups(addr, reg);
    };

    for_(int r = 0; r < nrows; r++)
    for (int cb = 0; cb < div_up(ncolumns, column_step); cb++) {
        load(r, cb);
        store(r, cb);
    }
}

void jit_copy_f32_t::generate() {
    preamble();

    const int row_block = conf_->os_block;
    const int row_tail = conf_->os % row_block;
    const int col_block = conf_->oc_block * conf_->nb_oc_blocking;
    const int col_tail = conf_->oc % col_block;
    src_stride = conf_->oc * typesize_data;
    tr_src_stride = conf_->LDB * typesize_data;
    src_batch_shift = src_stride * row_block;
    tr_src_batch_shift = tr_src_stride * row_block;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_row, ptr[param1 + GET_OFF(current_row_size)]);
    mov(reg_loop_col, ptr[param1 + GET_OFF(current_col_size)]);

    auto compute_batch = [=](int nrows, int ncolumns) {
        Label batch_loop;
        L(batch_loop);

        copy_block(nrows, ncolumns);
        add(reg_src, src_batch_shift);
        add(reg_tr_src, tr_src_batch_shift);

        sub(reg_loop_batch, 1);
        jnz(batch_loop, T_NEAR);
    };

    auto compute_rows = [=](int ncolumns) {
        Label row_done;
        if (row_tail > 0) {
            Label row_common;
            cmp(reg_loop_row, row_block);
            je(row_common, T_NEAR);

            compute_batch(row_tail, ncolumns);
            jmp(row_done, T_NEAR);

            L(row_common);
        }

        compute_batch(row_block, ncolumns);
        L(row_done);
    };

    Label col_done;
    if (col_tail > 0) {
        Label col_common;
        cmp(reg_loop_col, col_block);
        je(col_common, T_NEAR);

        compute_rows(col_tail);
        jmp(col_done, T_NEAR);

        L(col_common);
    }

    compute_rows(col_block);
    L(col_done);

    postamble();
}

struct jit_brgemm_trans_wei_f32_t : public jit_brgemm_trans_wei_t,
                                    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_wei_f32_t)

    jit_brgemm_trans_wei_f32_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_wei_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    enum { typesize = sizeof(float), transpose_size = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kCCCC = k4;
    opmask_t k0F0F = k5;
    opmask_t kF0F0 = k6;
    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_N = r10;
    reg64_t reg_loop_K = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;

    void transpose_16x16(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_f32_t::transpose_16x16(int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto load = [=](int i) {
        auto src_load = src_zmm(i);
        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < transpose_size;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                if (src_idx1 < nrows)
                    load(src_idx1);
                else
                    vpxord(src_zmm(src_idx1), src_zmm(src_idx1),
                            src_zmm(src_idx1));
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next) load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);

            if (next_src_idx1 < nrows && load_next) load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            int select_half = (i < 2) ? 0 : 2;
            int src_idx0 = base_idx + i + select_half + 0;
            int src_idx2 = src_idx0 + 2;

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx2);
            auto src0 = src_zmm(src_idx0);
            auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            valignd(tmp1, src2, src2, 0xe);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i;
            int src_idx4 = src_idx0 + 4;

            auto tmp0 = tmp_zmm(src_idx0);
            auto src0 = src_zmm(src_idx0);
            auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [=]() {
        // swap 8
        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_trans_wei_f32_t::generate() {
    preamble();
    assert(conf_->oc_block % transpose_size == 0);
    int fwd_ic_block = conf_->simd_w;
    int fwd_oc_block = 0;
    switch (conf_->wei_tag) {
        case OI16i64o:
        case OIw16i64o:
        case OIhw16i64o:
        case OIdhw16i64o:
        case OI8i64o2i:
        case OIw8i64o2i:
        case OIhw8i64o2i:
        case OIdhw8i64o2i:
        case OI16i64o2i:
        case OIw16i64o2i:
        case OIhw16i64o2i:
        case OIdhw16i64o2i: fwd_oc_block = 4 * conf_->simd_w; break;
        case OI16i32o:
        case OIw16i32o:
        case OIhw16i32o:
        case OIdhw16i32o:
        case OI8i32o2i:
        case OIw8i32o2i:
        case OIhw8i32o2i:
        case OIdhw8i32o2i:
        case OI16i32o2i:
        case OIw16i32o2i:
        case OIhw16i32o2i:
        case OIdhw16i32o2i: fwd_oc_block = 2 * conf_->simd_w; break;
        default: fwd_oc_block = conf_->simd_w;
    };

    int oc_tail = conf_->K_tail % transpose_size;
    int ic_block = conf_->ic_block;
    int ic_tail = conf_->N_tail % transpose_size;
    src_stride = fwd_oc_block * typesize;
    tr_src_stride = ic_block * typesize;
    dim_t N_src_shift = conf_->kd * conf_->kh * conf_->kw * fwd_ic_block
            * fwd_oc_block * typesize;
    dim_t N_tr_src_shift = conf_->simd_w * typesize;
    dim_t K_src_shift = conf_->simd_w * typesize;
    dim_t K_tr_src_shift = conf_->ic_block * conf_->simd_w * typesize;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    auto compute_N = [=](bool is_oc_tail) {
        mov(reg_loop_N, ptr[param1 + GET_OFF(current_N)]);
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        Label N_loop, N_loop_tail;

        cmp(reg_loop_N, transpose_size);
        jl(N_loop_tail, T_NEAR);

        L(N_loop);

        transpose_16x16(transpose_size, is_oc_tail ? oc_tail : transpose_size);
        add(reg_src, N_src_shift);
        add(reg_tr_src, N_tr_src_shift);

        sub(reg_loop_N, transpose_size);
        cmp(reg_loop_N, transpose_size);
        jge(N_loop, T_NEAR);

        L(N_loop_tail);
        if (ic_tail > 0) {
            Label N_loop_done;
            cmp(reg_loop_N, 0);
            jle(N_loop_done, T_NEAR);
            transpose_16x16(ic_tail, is_oc_tail ? oc_tail : transpose_size);
            L(N_loop_done);
        }
    };

    Label K_loop, K_tail;
    if (oc_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    L(K_loop);
    compute_N(false);
    add(reg_src_base, K_src_shift);
    add(reg_tr_src_base, K_tr_src_shift);

    sub(reg_loop_K, transpose_size);
    cmp(reg_loop_K, transpose_size);
    jge(K_loop, T_NEAR);

    L(K_tail);
    if (oc_tail > 0) {
        Label K_loop_done;
        cmp(reg_loop_K, 0);
        jle(K_loop_done, T_NEAR);

        compute_N(true);
        L(K_loop_done);
    }

    postamble();
}

struct jit_brgemm_trans_wei_bf16_t : public jit_brgemm_trans_wei_t,
                                     public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_trans_wei_bf16_t)

    jit_brgemm_trans_wei_bf16_t(const jit_brgemm_primitive_conf_t *conf)
        : jit_brgemm_trans_wei_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum { typesize = sizeof(int16_t), transpose_size = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kTail = k7;

    reg64_t reg_src_base = rax;
    reg64_t reg_tr_src_base = rbx;

    reg64_t reg_src = r8;
    reg64_t reg_tr_src = r9;
    reg64_t reg_loop_N = r10;
    reg64_t reg_loop_K = r11;
    reg64_t reg_loop_batch = r12;
    reg64_t reg_tr_src_tmp = r13;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = r15;

    zmm v_abcdefgh_to_abefcdgh = zmm31;

    void transpose_16x16_vnni(int nrows, int ncolumns = transpose_size);
    void generate() override;
};

void jit_brgemm_trans_wei_bf16_t::transpose_16x16_vnni(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 8);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 8);
        return Zmm(8 + i);
    };

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto load = [=](int i) {
        auto src_load = src_zmm(i);
        if (ncolumns < transpose_size) {
            kmovw(kTail, (1 << ncolumns) - 1);
            src_load = src_zmm(i) | kTail | T_z;
        }
        vmovups(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (nrows < transpose_size) kmovw(kTail, (1 << nrows) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < transpose_size;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    for (int i = 0; i < 8; i++)
        load(i);

    for (int i = 0; i < 8; i++)
        vpshufb(src_zmm(i), src_zmm(i), v_abcdefgh_to_abefcdgh);

    for (int i = 0; i < 2; i++) {
        vpunpcklqdq(tmp_zmm(2 * i + 0), src_zmm(2 * i), src_zmm(2 * i + 1));
        vpunpckhqdq(tmp_zmm(2 * i + 1), src_zmm(2 * i), src_zmm(2 * i + 1));
    }

    for (int i = 0; i < 2; i++) {
        vpunpcklqdq(
                src_zmm(2 * i + 0), src_zmm(4 + 2 * i), src_zmm(4 + 2 * i + 1));
        vpunpckhqdq(
                src_zmm(2 * i + 1), src_zmm(4 + 2 * i), src_zmm(4 + 2 * i + 1));
    }

    for (int i = 0; i < 2; i++) {
        vshufi32x4(src_zmm(4 + 0 + i), tmp_zmm(i), tmp_zmm(2 + i), 0x88);
        vshufi32x4(src_zmm(4 + 2 + i), tmp_zmm(i), tmp_zmm(2 + i), 0xdd);
    }

    for (int i = 0; i < 2; i++) {
        vshufi32x4(tmp_zmm(0 + i), src_zmm(i), src_zmm(2 + i), 0x88);
        vshufi32x4(tmp_zmm(2 + i), src_zmm(i), src_zmm(2 + i), 0xdd);
    }

    for (int i = 0; i < 4; i++)
        vshufi32x4(src_zmm(i), src_zmm(4 + i), tmp_zmm(i), 0x88);

    for (int i = 0; i < 4; i++)
        vshufi32x4(src_zmm(4 + i), src_zmm(4 + i), tmp_zmm(i), 0xdd);

    for (int i = 0; i < 8; i++)
        store(src_zmm(i), i);
}

void jit_brgemm_trans_wei_bf16_t::generate() {
    preamble();
    int fwd_oc_block = 0;
    switch (conf_->wei_tag) {
        case OI16i64o:
        case OIw16i64o:
        case OIhw16i64o:
        case OIdhw16i64o:
        case OI8i64o2i:
        case OIw8i64o2i:
        case OIhw8i64o2i:
        case OIdhw8i64o2i:
        case OI16i64o2i:
        case OIw16i64o2i:
        case OIhw16i64o2i:
        case OIdhw16i64o2i: fwd_oc_block = 4 * conf_->simd_w; break;
        case OI16i32o:
        case OIw16i32o:
        case OIhw16i32o:
        case OIdhw16i32o:
        case OI8i32o2i:
        case OIw8i32o2i:
        case OIhw8i32o2i:
        case OIdhw8i32o2i:
        case OI16i32o2i:
        case OIw16i32o2i:
        case OIhw16i32o2i:
        case OIdhw16i32o2i: fwd_oc_block = 2 * conf_->simd_w; break;
        default: fwd_oc_block = conf_->simd_w;
    };

    int oc_tail = conf_->K_tail % transpose_size;
    int ic_block = conf_->ic_block;
    int ic_tail = conf_->N_tail % transpose_size;
    src_stride = 2 * fwd_oc_block * typesize;
    tr_src_stride = 2 * ic_block * typesize;
    dim_t N_src_shift = conf_->simd_w * fwd_oc_block * typesize;
    dim_t N_tr_src_shift = 2 * conf_->simd_w * typesize;
    dim_t K_src_shift = 2 * conf_->simd_w * typesize;
    dim_t K_tr_src_shift = conf_->ic_block * conf_->simd_w * typesize;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_loop_batch, ptr[param1 + GET_OFF(current_gemm_batch)]);
    mov(reg_loop_K, ptr[param1 + GET_OFF(current_K)]);

    alignas(64) static constexpr const int32_t abcdefgh_to_abefcdgh[16]
            = {0x05040100, 0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100,
                    0x07060302, 0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302,
                    0x0d0c0908, 0x0f0e0b0a, 0x05040100, 0x07060302, 0x0d0c0908,
                    0x0f0e0b0a};

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(v_abcdefgh_to_abefcdgh, (const int64_t *)abcdefgh_to_abefcdgh);
    auto compute_N = [=](bool is_oc_tail) {
        mov(reg_src, reg_src_base);
        mov(reg_tr_src, reg_tr_src_base);
        mov(reg_loop_N, ptr[param1 + GET_OFF(current_N)]);

        Label N_loop, N_loop_tail;
        cmp(reg_loop_N, transpose_size);
        jl(N_loop_tail, T_NEAR);

        L(N_loop);

        transpose_16x16_vnni(
                transpose_size, is_oc_tail ? oc_tail : transpose_size);
        add(reg_src, N_src_shift);
        add(reg_tr_src, N_tr_src_shift);

        sub(reg_loop_N, transpose_size);
        cmp(reg_loop_N, transpose_size);
        jge(N_loop, T_NEAR);

        L(N_loop_tail);
        if (ic_tail > 0) {
            Label N_loop_done;
            cmp(reg_loop_N, 0);
            jle(N_loop_done, T_NEAR);
            transpose_16x16_vnni(
                    ic_tail, is_oc_tail ? oc_tail : transpose_size);
            L(N_loop_done);
        }
    };

    Label K_loop, K_tail;
    if (oc_tail > 0) {
        cmp(reg_loop_K, transpose_size);
        jl(K_tail, T_NEAR);
    }

    L(K_loop);
    compute_N(false);
    add(reg_src_base, K_src_shift);
    add(reg_tr_src_base, K_tr_src_shift);

    sub(reg_loop_K, transpose_size);
    cmp(reg_loop_K, transpose_size);
    jge(K_loop, T_NEAR);

    L(K_tail);
    if (oc_tail > 0) {
        Label K_loop_done;
        cmp(reg_loop_K, 0);
        jle(K_loop_done, T_NEAR);
        compute_N(true);
        L(K_loop_done);
    }

    postamble();
}

struct jit_amx_ip_trans_diff_wei_to_vnni_t : public jit_amx_ip_trans_diff_wei,
                                             public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_ip_trans_diff_wei_to_vnni)

    jit_amx_ip_trans_diff_wei_to_vnni_t(const jit_brgemm_primitive_conf_t *jbgp,
            const int ext_ic_block, const int ext_oc_block)
        : jit_amx_ip_trans_diff_wei(jbgp, ext_ic_block, ext_oc_block) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    void generate() override;
};

void jit_amx_ip_trans_diff_wei_to_vnni_t::generate() {
    const int typesize_out = 2;
    const int typesize_acc = 4;
    const int simd_w = 16;

    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;

    const reg64_t &reg_output = r15;
    const reg64_t &reg_input = r14;
    const reg64_t &reg_prm_table = r13;
    const reg64_t &reg_last_ic_block = r12;
    const reg64_t &reg_last_oc_block = r11;
    const reg32_t &regw_tmp = r10d;

    const Xbyak::Zmm &zmm_idx = Xbyak::Zmm(31);
    auto get_zmm_src = [&](int ic) { return Xbyak::Zmm(ic % 8); };

    Xbyak::Label prm_table;
    Xbyak::Label skip_oc_tail, to_exit;

    Xbyak::Opmask load_mask = k4;

    int tail_mask = (jbgp_->N_tail % simd_w)
            ? (1 << (jbgp_->N_tail % simd_w)) - 1
            : 0xffff;
    auto kmovw = [=](Xbyak::Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    auto reorder_oc_block = [&](int icb, int ic_block, bool is_oc_tail) {
        // INP:      [64i][No]         : FP32
        // OUT: [OCB][ICB][16i][No][2i]: BF16
        if (ic_block <= 0) return;

        dim_t inp_icb_offset = typesize_acc
                * (icb * ext_ic_block_ * jbgp_->oc_block); // Internal
        dim_t out_icb_offset = typesize_out
                * (icb * div_up(ext_ic_block_, 2) * ext_oc_block_
                        * 2); // External

        const int oc_padded = rnd_up(jbgp_->oc, jbgp_->oc_block);
        const int oc_padded_ext = rnd_up(jbgp_->oc, ext_oc_block_);

        bool tailing_done = false;
        for (int oc = 0; oc < jbgp_->oc_block; oc += simd_w) {
            int ext_oc = oc % ext_oc_block_;
            int ext_ocb = oc / ext_oc_block_;
            dim_t ext_ocb_offset = typesize_out
                    * (ext_ocb * div_up(jbgp_->ic, ext_ic_block_)
                            * div_up(ext_ic_block_, 2) * ext_oc_block_ * 2);
            if (is_oc_tail && oc_padded != oc_padded_ext
                    && oc + simd_w > ext_oc_block_)
                break;
            dim_t inp_offset = inp_icb_offset + typesize_acc * (oc); // Internal
            dim_t out_offset = out_icb_offset + typesize_out * (ext_oc * 2)
                    + ext_ocb_offset; // External
            kmovw(load_mask, 0xffff);
            if (is_oc_tail) {
                if (jbgp_->N_tail && (oc + simd_w) >= jbgp_->N_tail) {
                    if (tailing_done == false) {
                        kmovw(load_mask, tail_mask);
                        tailing_done = true;
                    } else {
                        auto zmm_src_0 = get_zmm_src(0);
                        vpxord(zmm_src_0, zmm_src_0, zmm_src_0);
                        for (int ic = 0; ic < ext_ic_block_ / 2; ic++) {
                            vmovups(ptr[reg_output + out_offset
                                            + typesize_out
                                                    * (ic * ext_oc_block_ * 2)],
                                    zmm_src_0);
                        }
                        continue;
                    }
                }
            }

            int ic = 0;
            for (; ic < ic_block / 2; ic++) {
                int ic1 = 2 * ic;
                int ic2 = 2 * ic + 1;

                auto zmm_src_0 = get_zmm_src(ic1);
                auto zmm_src_1 = get_zmm_src(ic2);

                vmovups(zmm_src_0 | load_mask | T_z,
                        ptr[reg_input + inp_offset
                                + typesize_acc * (ic1 * jbgp_->oc_block)]);
                vmovups(zmm_src_1 | load_mask | T_z,
                        ptr[reg_input + inp_offset
                                + typesize_acc * (ic2 * jbgp_->oc_block)]);

                vcvtne2ps2bf16(zmm_src_0, zmm_src_1, zmm_src_0);
                vpermw(zmm_src_0, zmm_idx, zmm_src_0);

                vmovups(ptr[reg_output + out_offset
                                + typesize_out * (ic * ext_oc_block_ * 2)],
                        zmm_src_0);
            }
            if (ic_block % 2) {
                int ic1 = 2 * ic;
                int ic2 = 2 * ic + 1;

                auto zmm_src_0 = get_zmm_src(ic1);
                auto zmm_src_1 = get_zmm_src(ic2);

                vmovups(zmm_src_0 | load_mask | T_z,
                        ptr[reg_input + inp_offset
                                + typesize_acc * (ic1 * jbgp_->oc_block)]);
                vpxord(zmm_src_1, zmm_src_1, zmm_src_1);

                vcvtne2ps2bf16(zmm_src_0, zmm_src_1, zmm_src_0);
                vpermw(zmm_src_0, zmm_idx, zmm_src_0);

                vmovups(ptr[reg_output + out_offset
                                + typesize_out * (ic * ext_oc_block_ * 2)],
                        zmm_src_0);
                ic++;
            }
            if (ic < ext_ic_block_ / 2) {
                auto zmm_src_0 = get_zmm_src(0);
                vpxord(zmm_src_0, zmm_src_0, zmm_src_0);
                for (; ic < ext_ic_block_ / 2; ic++) {
                    vmovups(ptr[reg_output + out_offset
                                    + typesize_out * (ic * ext_oc_block_ * 2)],
                            zmm_src_0);
                }
            }
        }
    };

    auto reorder_ic_block = [&](bool is_oc_tail, bool is_ic_tail) {
        int nb_ic = div_up(jbgp_->ic_block, ext_ic_block_);
        for (int icb = 0; icb < nb_ic; icb++) {
            int ic_0 = icb * ext_ic_block_;
            int ic_1 = (icb + 1) * ext_ic_block_;
            if (is_ic_tail) {
                int ext_ic_tail = (jbgp_->ic % ext_ic_block_)
                        ? (jbgp_->ic % ext_ic_block_)
                        : ext_ic_block_;
                if (jbgp_->M_tail && ic_0 >= jbgp_->M_tail) break;
                if (jbgp_->M_tail && ic_0 <= jbgp_->M_tail
                        && jbgp_->M_tail <= ic_1) {
                    reorder_oc_block(icb, ext_ic_tail, is_oc_tail);
                } else {
                    reorder_oc_block(icb, ext_ic_block_, is_oc_tail);
                }
            } else {
                reorder_oc_block(icb, ext_ic_block_, is_oc_tail);
            }
        }
    };

    auto reorder = [&](bool is_oc_tail) {
        Xbyak::Label skip_ic_tail, to_exit_1;

        cmp(reg_last_ic_block, 0);
        je(skip_ic_tail, T_NEAR);

        reorder_ic_block(is_oc_tail, true);
        jmp(to_exit, T_NEAR);

        L(skip_ic_tail);
        reorder_ic_block(is_oc_tail, false);

        L(to_exit_1);
    };

    preamble();

    mov(reg_input, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_output, ptr[abi_param1 + GET_OFF(dst)]);
    mov(reg_last_ic_block, ptr[abi_param1 + GET_OFF(last_ic_block)]);
    mov(reg_last_oc_block, ptr[abi_param1 + GET_OFF(last_oc_block)]);

    mov(reg_prm_table, prm_table);
    vmovups(zmm_idx, ptr[reg_prm_table]);

    cmp(reg_last_oc_block, 0);
    je(skip_oc_tail, T_NEAR);

    reorder(true);
    jmp(to_exit, T_NEAR);

    L(skip_oc_tail);
    reorder(false);

    L(to_exit);
    postamble();

    align(64);
    L(prm_table);
    const uint16_t prm_array[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    for (size_t i = 0; i < 32; ++i)
        dw(prm_array[i]);
}

#undef GET_OFF

status_t create_brgemm_trans_src(
        std::unique_ptr<jit_brgemm_trans_src_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (conf->prop_kind == dnnl_backward_weights
            && conf->src_dt == data_type::f32)
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_m_k_f32_t(conf)));
    else if (conf->prop_kind == dnnl_backward_weights
            && conf->src_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_brgemm_trans_m_k_bf16_t(conf)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

status_t create_brgemm_copy_to_coarse(
        std::unique_ptr<jit_brgemm_copy_to_coarse_t> &copy_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (conf->isa == avx512_core_bf16_amx_int8
            || conf->isa == avx512_core_bf16_amx_bf16)
        CHECK(safe_ptr_assign(copy_ker, new jit_brgemm_copy_to_coarse_t(conf)));
    else
        return status::invalid_arguments;

    return copy_ker->create_kernel();
}

status_t create_brgemm_trans_to_vnni(
        std::unique_ptr<jit_brgemm_trans_to_vnni_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf,
        jit_brgemm_trans_to_vnni_t::matrix_to_transform_t matrix_to_transform) {
    if (conf->prop_kind == dnnl_backward_weights
            && conf->dst_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_trans_to_vnni_t(conf, matrix_to_transform)));
    else if (conf->prop_kind == dnnl_backward_weights
            && conf->dst_dt == data_type::f32)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_copy_f32_t(conf, matrix_to_transform)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

status_t create_brgemm_trans_wei(
        std::unique_ptr<jit_brgemm_trans_wei_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf) {
    if (conf->prop_kind == dnnl_backward_data && conf->wei_dt == data_type::f32)
        CHECK(safe_ptr_assign(trans_ker, new jit_brgemm_trans_wei_f32_t(conf)));
    else if (conf->prop_kind == dnnl_backward_data
            && conf->wei_dt == data_type::bf16)
        CHECK(safe_ptr_assign(
                trans_ker, new jit_brgemm_trans_wei_bf16_t(conf)));
    else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

status_t create_brgemm_amx_ip_trans_wei(
        std::unique_ptr<jit_amx_ip_trans_diff_wei> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf, const int ext_ic_block,
        const int ext_oc_block) {
    if (conf->prop_kind == dnnl_backward_weights
            && conf->wei_dt == data_type::bf16) {
        CHECK(safe_ptr_assign(trans_ker,
                new jit_amx_ip_trans_diff_wei_to_vnni_t(
                        conf, ext_ic_block, ext_oc_block)));
    } else
        return status::invalid_arguments;

    return trans_ker->create_kernel();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
