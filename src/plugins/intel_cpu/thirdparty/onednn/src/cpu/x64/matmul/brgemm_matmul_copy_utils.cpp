/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_brgemm_matmul_copy_a_impl_t : public jit_brgemm_matmul_copy_a_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_impl_t)

    jit_brgemm_matmul_copy_a_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , typesize(conf_->a_dt_sz)
        , vnni_granularity(granularity_max / typesize)
        , k_step(bytes_in_zmm / typesize) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;
    using xmm = const Xbyak::Xmm;

    enum {
        granularity_max = 4,
        num_comp_acc = 8,
        k_loop_unroll = 16,
        bytes_in_zmm = 64,
    };
    const int typesize;
    const int vnni_granularity;
    const int k_step;

    dim_t src_stride = 0, tr_src_stride = 0;
    bool do_compute_compensation = false;

    opmask_t kTail_load = k7;
    opmask_t kTail_store = k6;
    opmask_t kTail_comp = k5;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_K_start = abi_not_param1;

    reg64_t reg_zp_comp_buf_ptr = rdx;
    reg64_t reg_zp_comp_res_ptr = rsi;

    reg64_t reg_M_blk = r9;
    reg64_t reg_K_blk = r10;
    reg64_t reg_batch = r11;
    reg64_t reg_aux_src = r12;
    reg64_t reg_aux_tr_src = r13;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_zp_ab_comp_ptr = imm_addr64;
    reg64_t reg_zp_b_neg_val_ptr = reg_K_blk;

    zmm zmm_comp_mul = zmm30;
    zmm zmm_comp_add = zmm31;

    // Allows to shift A data by 128 for s8s8 problem for AVX512 in copy
    // routine, not in compute kernel. It's disabled for now, as it
    // requires setting some hint to brgemm kerenel to avoid double shifting
    const bool allow_input_shift_for_s8s8 = false;

    Xbyak::Zmm get_zmm_comp_acc(int i) {
        assert(i >= 0 && i < num_comp_acc);
        return Xbyak::Zmm(i);
    }

    Xbyak::Zmm get_zmm_copy(int i) {
        assert(i >= 0 && i < k_loop_unroll);
        return Xbyak::Zmm(29 - i);
    }
    void reduce_compensation_across_accumulators(int num_accumulators);
    void copy_row(int ncolumns);
    void copy_K_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    void copy_M_loop(bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter);
    void generate() override;
};

void jit_brgemm_matmul_copy_a_impl_t::reduce_compensation_across_accumulators(
        int num_accumulators) {
    int num = num_accumulators;
    while (num > 1) {
        for (int i = 0; i < num / 2; i++) {
            const auto zmm_acc0 = get_zmm_comp_acc(i);
            const auto zmm_acc1 = get_zmm_comp_acc(div_up(num, 2) + i);
            vpaddd(zmm_acc0, zmm_acc0, zmm_acc1);
        }
        num = div_up(num, 2);
    }
}

void jit_brgemm_matmul_copy_a_impl_t::copy_K_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    MAYBE_UNUSED(is_K_tail);
    MAYBE_UNUSED(is_first_K_iter);
    MAYBE_UNUSED(is_last_K_iter);

    const int K_blk = is_K_tail ? conf_->K % conf_->K_blk
                                : nstl::min(conf_->K, conf_->K_blk);
    const int k_tail = K_blk % k_step;
    const int num_k_iters = K_blk / k_step;
    const int num_acc = utils::saturate(1, (int)num_comp_acc, num_k_iters);

    if (do_compute_compensation) {
        for (int i = 0; i < num_acc; i++) {
            const auto zmm_acc = get_zmm_comp_acc(i);
            vpxord(zmm_acc, zmm_acc, zmm_acc);
        }
    }

    auto maybe_compute_compensation = [=](int k_idx, zmm zmm_copy) {
        if (do_compute_compensation) {
            const auto zmm_comp_acc = get_zmm_comp_acc(k_idx % num_acc);
            if (conf_->src_dt == data_type::s8)
                vpdpbusd(zmm_comp_acc, zmm_comp_mul, zmm_copy);
            else
                vpdpbusd(zmm_comp_acc, zmm_copy, zmm_comp_mul);
        }
    };

    for (int kb = 0; kb < div_up(num_k_iters, k_loop_unroll); kb++) {
        int k_start = 0;
        int k_end = nstl::min(
                (int)k_loop_unroll, num_k_iters - kb * k_loop_unroll);
        for (int k = k_start; k < k_end; k++) {
            const int k_idx = kb * k_loop_unroll + k;
            const size_t offset = (size_t)k_idx * k_step * typesize;
            vmovdqu8(get_zmm_copy(k), EVEX_compress_addr(reg_src, offset));

            maybe_compute_compensation(k_idx, get_zmm_copy(k));
        }
        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
            for (int k = k_start; k < k_end; k++)
                vpaddb(get_zmm_copy(k), get_zmm_copy(k), zmm_comp_add);
        }

        for (int k = k_start; k < k_end; k++) {
            const size_t offset
                    = ((size_t)kb * k_loop_unroll + k) * k_step * typesize;
            vmovdqu8(EVEX_compress_addr(reg_tr_src, offset), get_zmm_copy(k));
        }
    }

    if (k_tail > 0) {
        const auto kmovq = [=](Opmask k, size_t q) {
            mov(regq_tmp, q);
            jit_generator::kmovq(k, regq_tmp);
        };

        const size_t tail_mask_load
                = size_t(((size_t)1 << (typesize * k_tail)) - 1);
        kmovq(kTail_load, tail_mask_load);
        const int k_tail_st = rnd_up(k_tail, vnni_granularity);
        const size_t tail_mask_store = k_tail_st == k_step
                ? 0xffffffffffffffff
                : size_t(((size_t)1 << (typesize * k_tail_st)) - 1);
        kmovq(kTail_store, tail_mask_store);

        auto zmm_tail = get_zmm_copy(0) | kTail_load | T_z;
        vmovdqu8(zmm_tail,
                EVEX_compress_addr(reg_src, num_k_iters * k_step * typesize));

        maybe_compute_compensation(0, get_zmm_copy(0));

        if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required)
            vpaddb(get_zmm_copy(0), get_zmm_copy(0), zmm_comp_add);

        vmovdqu8(
                EVEX_compress_addr(reg_tr_src, num_k_iters * k_step * typesize),
                get_zmm_copy(0) | kTail_store);
    }

    if (do_compute_compensation) {
        reduce_compensation_across_accumulators(num_acc);

        const auto addr_buf = zword[reg_zp_comp_buf_ptr];
        if (!is_first_K_iter)
            vpaddd(get_zmm_comp_acc(0), get_zmm_comp_acc(0), addr_buf);
        if (!is_last_K_iter) {
            vmovups(addr_buf, get_zmm_comp_acc(0));
            return;
        }

        // is_last_K_iter == true: we need to reduce values within acc
        // register, add mixed ab_compensation component if any, multiply
        // it by negative zp_b_value and finally store the reslt

        // step 1: reduce values within acc register
        const auto ymm_red0 = ymm(get_zmm_comp_acc(0).getIdx());
        const auto ymm_red1 = ymm(get_zmm_comp_acc(1).getIdx());
        vextracti64x4(ymm_red1, get_zmm_comp_acc(0), 1);
        vphaddd(ymm_red0, ymm_red0, ymm_red1);
        vpxord(ymm_red1, ymm_red1, ymm_red1);
        vphaddd(ymm_red0, ymm_red0, ymm_red1);
        vphaddd(ymm_red0, ymm_red0, ymm_red1);
        const auto xmm_red1 = xmm(ymm_red1.getIdx());
        vextractf128(xmm_red1, ymm_red0, 1);
        vpaddd(ymm_red0, ymm_red0, ymm_red1);

        // step 2: add -K * zp_a_val as mixed ab_compensation component
        if (conf_->src_zp_type != brgemm_broadcast_t::none) {
            assert(conf_->src_zp_type == brgemm_broadcast_t::per_tensor);
            reg64_t reg_zp_ab_comp_ptr = imm_addr64;
            mov(reg_zp_ab_comp_ptr, ptr[param1 + GET_OFF(zp_ab_comp_ptr)]);

            const auto addr_ab_comp = zword_b[reg_zp_ab_comp_ptr];
            const auto zmm_res = get_zmm_comp_acc(0) | kTail_comp;
            vpaddd(zmm_res, get_zmm_comp_acc(0), addr_ab_comp);
        }

        // step 3: multiply by zp_b_val
        mov(reg_zp_b_neg_val_ptr, ptr[param1 + GET_OFF(zp_b_neg_value_ptr)]);
        const auto zmm_zp_b_neg_val = get_zmm_comp_acc(1);
        vbroadcastss(zmm_zp_b_neg_val, ptr[reg_zp_b_neg_val_ptr]);
        vpmulld(get_zmm_comp_acc(0), get_zmm_comp_acc(0), zmm_zp_b_neg_val);

        // step 4: store the final result value
        vmovups(ptr[reg_zp_comp_res_ptr], get_zmm_comp_acc(0) | kTail_comp);
    }
}

void jit_brgemm_matmul_copy_a_impl_t::copy_M_loop(
        bool is_K_tail, bool is_first_K_iter, bool is_last_K_iter) {

    if (do_compute_compensation) {
        mov(imm_addr64, 1);
        vpbroadcastb(zmm_comp_mul, imm_addr64.cvt8());
        if (!(is_first_K_iter && is_last_K_iter))
            mov(reg_zp_comp_buf_ptr,
                    ptr[param1 + GET_OFF(zp_b_compensation_buffer_ptr)]);

        if (is_last_K_iter) {
            mov(reg_zp_comp_res_ptr,
                    ptr[param1 + GET_OFF(zp_a_compensation_result_ptr)]);
            const auto kmovw = [=](Opmask k, size_t q) {
                mov(regq_tmp, q);
                jit_generator::kmovw(k, imm_addr64.cvt32());
            };
            kmovw(kTail_comp, 1);
        }
    }

    Label loop_M;
    L(loop_M);

    copy_K_loop(is_K_tail, is_first_K_iter, is_last_K_iter);

    add(reg_src, src_stride);
    add(reg_tr_src, tr_src_stride);
    if (do_compute_compensation) {
        // shift comp pointers
        if (!(is_first_K_iter && is_last_K_iter))
            add(reg_zp_comp_buf_ptr, sizeof(int32_t) * 16);
        if (is_last_K_iter) add(reg_zp_comp_res_ptr, sizeof(int32_t));
    }

    dec(reg_M_blk);
    jnz(loop_M, T_NEAR);
}

void jit_brgemm_matmul_copy_a_impl_t::generate() {
    preamble();
    src_stride = conf_->K * typesize;
    const dim_t LDA = conf_->use_buffer_a_tail_only ? (dim_t)conf_->wei_k_blk
                                                    : conf_->LDA;
    tr_src_stride = LDA * typesize;
    do_compute_compensation = conf_->has_zero_point_b;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_blk, ptr[param1 + GET_OFF(current_K_blk)]);
    mov(reg_M_blk, ptr[param1 + GET_OFF(current_M_blk)]);

    if (allow_input_shift_for_s8s8 && conf_->s8s8_compensation_required) {
        mov(imm_addr64, 128);
        vpbroadcastb(zmm_comp_add, imm_addr64.cvt8());
    }

    auto copy_body = [=](bool is_first_K_iter, bool is_last_K_iter) {
        Label copy_body_done;
        // might be different from conf_->K_tail
        const dim_t K_blk_tail
                = conf_->K_tail > 0 ? conf_->K % conf_->K_blk : 0;
        if (K_blk_tail > 0) {
            Label not_K_tail;
            cmp(reg_K_blk, K_blk_tail);
            jne(not_K_tail, T_NEAR);
            copy_M_loop(true, is_first_K_iter, is_last_K_iter);
            jmp(copy_body_done, T_NEAR);

            L(not_K_tail);
        }

        copy_M_loop(false, is_first_K_iter, is_last_K_iter);
        L(copy_body_done);
    };

    Label done;
    if (do_compute_compensation) {
        assert(conf_->wei_zp_type == brgemm_broadcast_t::per_tensor);

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            copy_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            copy_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        copy_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }
    copy_body(false, false);
    L(done);

    postamble();
}

struct jit_brgemm_matmul_copy_a_transposed_impl_t
    : public jit_brgemm_matmul_copy_a_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_a_transposed_impl_t)

    jit_brgemm_matmul_copy_a_transposed_impl_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_a_t(conf)
        , typesize(conf_->a_dt_sz)
        , src_stride(conf_->M * typesize)
        , dst_stride(conf_->LDA * typesize)
        , m_loop_src_shift(columns_step * typesize)
        , m_loop_dst_shift(columns_step * dst_stride)
        , k_loop_src_shift(rows_step * src_stride)
        , k_loop_dst_shift(rows_step * typesize)
        , is_f32(everyone_is(data_type::f32, conf_->src_dt, conf_->wei_dt)) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;

    const size_t typesize;
    const int rows_step = 16;
    const int columns_step = rows_step;
    const dim_t src_stride, dst_stride;
    const dim_t m_loop_src_shift;
    const dim_t m_loop_dst_shift;
    const dim_t k_loop_src_shift;
    const dim_t k_loop_dst_shift;
    const bool is_f32;

    opmask_t kFFFF = k1;
    opmask_t k3333 = k1;
    opmask_t k5555 = k2;
    opmask_t kAAAA = k3;
    opmask_t kAA = k4;
    opmask_t kCCCC = k4;
    opmask_t k55 = k5;
    opmask_t k0F0F = k5;
    opmask_t kCC = k6;
    opmask_t kF0F0 = k6;
    opmask_t k33 = k7;
    opmask_t kTail = is_f32 ? k7 : k1;

    reg32_t regw_tmp = r15d;
    reg64_t reg_k_src = r14;
    reg64_t reg_k_dst = r13;
    reg64_t reg_m_src = r12;
    reg64_t reg_m_dst = r11;
    reg64_t reg_loop_k = rax;
    reg64_t reg_loop_m = rbx;
    reg64_t imm_addr64 = rdx;

    Xbyak::Zmm vidx1 = zmm31;
    Xbyak::Zmm vidx2 = zmm30;
    Xbyak::Zmm vidx3 = zmm29;
    Xbyak::Zmm vidx4 = zmm28;
    Xbyak::Zmm vidx5 = zmm27;
    Xbyak::Zmm zmm_tmp = zmm26;

    void transpose_f32(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void transpose_bf16(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void deploy_transpose(reg64_t dst, reg64_t src, int nrows, int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_a_transposed_impl_t::transpose_bf16(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;

    auto src_zmm = [=](int i) { return Zmm(i); };

    auto src_ymm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    };

    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    auto store = [=](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovdqu16(addr, r | kTail);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovd(kFFFF, load_mask);

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

    for (int i = rnd_up(nrows, 2); i < rows_step; i += 2) {
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

    auto get_vec_idx = [=](int col_idx) {
        assert(col_idx < columns_step && col_idx >= 0);
        const int blk_sz = 4;
        const int blk_idx = col_idx / blk_sz;
        const int idx_within_blk = col_idx % blk_sz;

        // 0 1 2 3 -> 0 2 1 3
        const int mapped_blk_idx = 2 * blk_idx - (blk_idx / 2) * 3;
        // 0 1 2 3 -> 1 0 3 2
        const int mapped_idx_within_blk
                = idx_within_blk + 1 - 2 * (idx_within_blk % 2);
        return blk_sz * mapped_blk_idx + mapped_idx_within_blk;
    };
    const int columns_to_store = rnd_up(nrows, 2);
    const int store_mask = columns_to_store < rows_step
            ? (1 << columns_to_store) - 1
            : 0xffff;
    kmovd(kTail, store_mask);

    for (int col_idx = 0; col_idx < ncolumns; col_idx++)
        store(src_zmm(get_vec_idx(col_idx)), col_idx);
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::transpose_f32(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= rows_step && ncolumns >= 0
            && ncolumns <= columns_step);
    if (!nrows) return;

    auto kmovw = [=](Opmask k, size_t q) {
        mov(regw_tmp, q);
        jit_generator::kmovw(k, regw_tmp);
    };

    const int load_mask
            = ncolumns < columns_step ? (1 << ncolumns) - 1 : 0xffff;
    kmovw(kTail, load_mask);

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [=](int i) {
        if (i < nrows)
            vmovups(src_zmm(i) | kTail | T_z,
                    EVEX_compress_addr(src, i * src_stride));
        else
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
    };

    auto store = [=](Zmm r, int i) {
        auto addr = EVEX_compress_addr(dst, i * dst_stride);
        vmovups(addr, r | kTail);
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
                load(src_idx1);
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
        const int store_mask = nrows < rows_step ? (1 << nrows) - 1 : 0xffff;
        kmovw(kTail, store_mask);

        // swap 8
        for (int i = 0; i < nstl::min(8, ncolumns); i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
        }

        for (int i = 0; i < nstl::max(0, ncolumns - 8); i++) {
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

void jit_brgemm_matmul_copy_a_transposed_impl_t::deploy_transpose(
        reg64_t dst, reg64_t src, int nrows, int ncolumns) {
    if (is_f32)
        transpose_f32(dst, src, nrows, ncolumns);
    else
        transpose_bf16(dst, src, nrows, ncolumns);
}

void jit_brgemm_matmul_copy_a_transposed_impl_t::generate() {

    // only bf16 and f32 supported for now
    if (!one_of(conf_->src_dt, data_type::bf16, data_type::f32)) return;

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

    const int k_block_tail = conf_->K_blk % rows_step;
    const int last_k_block_tail = (conf_->K % conf_->K_blk) % rows_step;
    const int m_block_tail = conf_->M_blk % columns_step;
    const int last_m_block_tail = conf_->M_tail % columns_step;

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    if (is_f32) {
        kmovw(k3333, 0x3333); // 0011001100110011
        kmovw(k5555, 0x5555); // 0101010101010101
        kmovw(kAAAA, 0xaaaa); // 1010101010101010
        kmovw(kCCCC, 0xcccc); // 1100110011001100
        kmovw(k0F0F, 0x0f0f); // 0000111100001111
        kmovw(kF0F0, 0xf0f0); // 1111000011110000
    } else {
        kmovw(kFFFF, 0xffff);
        kmovw(k5555, 0x5555);
        kmovw(kAAAA, 0xaaaa);
        kmovw(kAA, 0xaa);
        kmovw(k55, 0x55);
        kmovw(kCC, 0xcc);
        kmovw(k33, 0x33);
    }

    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [=](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa32(z, ptr[imm_addr64]);
    };

    if (!is_f32) {
        vmovdqa64(vidx1, idx1);
        vmovdqa64(vidx2, idx2);
        vmovdqa32(vidx3, idx3);
        vmovdqa32(vidx4, idx4);
        vmovdqa32(vidx5, (const int32_t *)idx5);
    }

    auto compute_m_loop = [&](reg64_t &reg_base, reg64_t &reg_tr_base,
                                  int nrows) {
        mov(reg_loop_m, ptr[param1 + GET_OFF(current_M_blk)]);
        mov(reg_m_src, reg_base);
        mov(reg_m_dst, reg_tr_base);

        Label m_loop_tail_or_done, m_loop, compute_m_loop_done;
        cmp(reg_loop_m, columns_step);
        jl(m_loop_tail_or_done, T_NEAR);

        L(m_loop);
        {
            deploy_transpose(reg_m_dst, reg_m_src, nrows, columns_step);
            add(reg_m_src, m_loop_src_shift);
            add(reg_m_dst, m_loop_dst_shift);
        }
        sub(reg_loop_m, columns_step);
        cmp(reg_loop_m, columns_step);
        jge(m_loop, T_NEAR);

        if (m_block_tail > 0 || last_m_block_tail > 0)
            jz(compute_m_loop_done, T_NEAR);

        L(m_loop_tail_or_done);

        if (m_block_tail > 0) {
            Label m_block_tail_done;
            cmp(reg_loop_m, m_block_tail);
            jne(m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows, m_block_tail);
            jmp(compute_m_loop_done, T_NEAR);

            L(m_block_tail_done);
        }
        if (last_m_block_tail > 0 && last_m_block_tail != m_block_tail) {
            Label last_m_block_tail_done;
            cmp(reg_loop_m, last_m_block_tail);
            jne(last_m_block_tail_done, T_NEAR);

            deploy_transpose(reg_m_dst, reg_m_src, nrows, last_m_block_tail);
            jmp(compute_m_loop_done, T_NEAR);

            L(last_m_block_tail_done);
        }

        L(compute_m_loop_done);
    };

    auto compute_k_loop = [&]() {
        mov(reg_k_src, ptr[param1 + GET_OFF(src)]);
        mov(reg_k_dst, ptr[param1 + GET_OFF(tr_src)]);
        mov(reg_loop_k, ptr[param1 + GET_OFF(current_K_blk)]);

        Label k_tail_or_done, k_loop, compute_k_loop_done;
        cmp(reg_loop_k, rows_step);
        jl(k_tail_or_done, T_NEAR);

        L(k_loop);
        {
            compute_m_loop(reg_k_src, reg_k_dst, rows_step);
            add(reg_k_src, k_loop_src_shift);
            add(reg_k_dst, k_loop_dst_shift);
        }
        sub(reg_loop_k, rows_step);
        cmp(reg_loop_k, rows_step);
        jge(k_loop, T_NEAR);

        if (k_block_tail > 0 || last_k_block_tail > 0)
            jz(compute_k_loop_done, T_NEAR);

        L(k_tail_or_done);

        if (k_block_tail > 0) {
            Label k_block_tail_done;
            cmp(reg_loop_k, k_block_tail);
            jne(k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(k_block_tail_done);
        }
        if (last_k_block_tail > 0 && last_k_block_tail != k_block_tail) {
            Label last_k_block_tail_done;
            cmp(reg_loop_k, last_k_block_tail);
            jne(last_k_block_tail_done, T_NEAR);

            compute_m_loop(reg_k_src, reg_k_dst, last_k_block_tail);
            jmp(compute_k_loop_done, T_NEAR);

            L(last_k_block_tail_done);
        }

        L(compute_k_loop_done);
    };

    compute_k_loop();

    postamble();
}

struct jit_brgemm_matmul_copy_b_int8_t : public jit_brgemm_matmul_copy_b_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_int8_t)

    jit_brgemm_matmul_copy_b_int8_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;

    enum { typesize = sizeof(int8_t), k_blk_step = 4, n_blk_step = 64 };
    dim_t src_stride = 0, tr_src_stride = 0;
    bool is_amx = false;
    bool do_compute_compensation = false;

    opmask_t kTail = k7;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;
    reg64_t reg_comp_ptr = rdx;
    reg64_t reg_zp_comp_ptr = r11;
    reg64_t reg_zp_a_neg_val_ptr = r12;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg64_t regq_tmp = r14;
    reg64_t imm_addr64 = r15;

    zmm vreg_idx_lo_256 = zmm26;
    zmm vreg_idx_hi_256 = zmm27;
    zmm vreg_idx_lo_128 = zmm28;
    zmm vreg_idx_hi_128 = zmm29;
    zmm zmm_comp_mul = zmm30;
    zmm zmm_zero = zmm31;

    Xbyak::Zmm get_comp_acc(int i) { return Xbyak::Zmm(25 - i); }
    Xbyak::Zmm get_zmm_zp_comp_res(int i) { return get_comp_acc(i); }
    Xbyak::Zmm get_zmm_oscale_comp_res(int i) { return Xbyak::Zmm(i); }
    void copy_4x64_vnni_avx512_core(int nrows, int ncolumns);
    void copy_4x64_vnni_amx(int nrows, int ncolumns);
    void copy_4x64_vnni(int nrows, int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_int8_t::copy_4x64_vnni(int nrows, int ncolumns) {
    if (is_amx)
        copy_4x64_vnni_amx(nrows, ncolumns);
    else
        copy_4x64_vnni_avx512_core(nrows, ncolumns);
}

void jit_brgemm_matmul_copy_b_int8_t::copy_4x64_vnni_amx(
        int nrows, int ncolumns) {
    auto kmovq = [=](Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
    };

    const auto tail_mask = size_t(((size_t)1 << ncolumns) - 1);
    if (ncolumns < n_blk_step) kmovq(kTail, tail_mask);

    const int blk_sz = 6;
    const int max_unroll = (do_compute_compensation ? 21 : 25) / blk_sz;
    auto get_zmm = [=](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = blk_sz * blk + idx;
        assert(reg_idx >= 0 && reg_idx < 32);
        return zmm(reg_idx);
    };

    auto load = [=](int blk, int i) {
        auto src_reg = get_zmm(blk, i % k_blk_step);
        auto src_load = ncolumns < n_blk_step ? src_reg | kTail | T_z : src_reg;
        vmovdqu8(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step); kb++)
    for (int k = 0;
            k < nstl::min(max_unroll,
                    div_up(nrows - kb * max_unroll * k_blk_step, k_blk_step));
            k++) {
        const int row_start = (kb * max_unroll + k) * k_blk_step;
        const int row_end = nstl::min(row_start + k_blk_step, nrows);

        for (int i = row_start; i < row_end; i++)
            load(k, i);
        if (row_end == nrows && nrows % k_blk_step > 0) {
            for (int i = nrows; i < rnd_up(nrows, k_blk_step); i++) {
                auto src_reg = get_zmm(k, i % k_blk_step);
                vpxord(src_reg, src_reg, src_reg);
            }
        }

        vmovups(get_zmm(k, 4), vreg_idx_lo_256);
        vpermi2b(get_zmm(k, 4), get_zmm(k, 0), get_zmm(k, 2));
        vmovups(get_zmm(k, 5), vreg_idx_hi_256);
        vpermi2b(get_zmm(k, 5), get_zmm(k, 0), get_zmm(k, 2));
        vmovups(get_zmm(k, 0), vreg_idx_lo_256);
        vpermi2b(get_zmm(k, 0), get_zmm(k, 1), get_zmm(k, 3));
        vmovups(get_zmm(k, 2), vreg_idx_hi_256);
        vpermi2b(get_zmm(k, 2), get_zmm(k, 1), get_zmm(k, 3));

        vmovups(get_zmm(k, 1), vreg_idx_lo_128);
        vpermi2b(get_zmm(k, 1), get_zmm(k, 4), get_zmm(k, 0));
        dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride;
        vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base), get_zmm(k, 1));
        if (do_compute_compensation)
            vpdpbusd(get_comp_acc(0), zmm_comp_mul, get_zmm(k, 1));

        if (ncolumns > 16) {
            vmovups(get_zmm(k, 3), vreg_idx_hi_128);
            vpermi2b(get_zmm(k, 3), get_zmm(k, 4), get_zmm(k, 0));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    get_zmm(k, 3));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(1), zmm_comp_mul, get_zmm(k, 3));
        } else if (conf_->wei_n_blk > 16) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    zmm_zero);
        }

        if (ncolumns > 32) {
            vmovups(get_zmm(k, 4), vreg_idx_lo_128);
            vpermi2b(get_zmm(k, 4), get_zmm(k, 5), get_zmm(k, 2));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    get_zmm(k, 4));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(2), zmm_comp_mul, get_zmm(k, 4));
        } else if (conf_->wei_n_blk > 32) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    zmm_zero);
        }

        if (ncolumns > 48) {
            vmovups(get_zmm(k, 0), vreg_idx_hi_128);
            vpermi2b(get_zmm(k, 0), get_zmm(k, 5), get_zmm(k, 2));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    get_zmm(k, 0));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(3), zmm_comp_mul, get_zmm(k, 0));
        } else if (conf_->wei_n_blk > 48) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    zmm_zero);
        }
    }
}

void jit_brgemm_matmul_copy_b_int8_t::copy_4x64_vnni_avx512_core(
        int nrows, int ncolumns) {
    auto kmovq = [=](Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
    };

    const auto tail_mask = size_t(((size_t)1 << ncolumns) - 1);
    if (ncolumns < n_blk_step) kmovq(kTail, tail_mask);

    const int blk_sz = 6;
    const int max_unroll = (do_compute_compensation ? 21 : 25) / blk_sz;
    auto get_zmm = [=](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = blk_sz * blk + idx;
        assert(reg_idx >= 0 && reg_idx < 32);
        return zmm(reg_idx);
    };
    auto load = [=](int blk, int i) {
        auto src_reg = get_zmm(blk, i % k_blk_step);
        auto src_load = ncolumns < n_blk_step ? src_reg | kTail | T_z : src_reg;
        vmovdqu8(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    for_(int kb = 0; kb < div_up(nrows, max_unroll * k_blk_step); kb++)
    for (int k = 0;
            k < nstl::min(max_unroll,
                    div_up(nrows - kb * max_unroll * k_blk_step, k_blk_step));
            k++) {
        const int row_start = (kb * max_unroll + k) * k_blk_step;
        const int row_end = nstl::min(row_start + k_blk_step, nrows);

        for (int i = row_start; i < row_end; i++)
            load(k, i);
        if (row_end == nrows && nrows % k_blk_step > 0) {
            for (int i = nrows; i < rnd_up(nrows, k_blk_step); i++) {
                auto src_reg = get_zmm(k, i % k_blk_step);
                vpxord(src_reg, src_reg, src_reg);
            }
        }

        vpunpcklbw(get_zmm(k, 4), get_zmm(k, 0), get_zmm(k, 1));
        vpunpckhbw(get_zmm(k, 5), get_zmm(k, 0), get_zmm(k, 1));
        vpunpcklbw(get_zmm(k, 0), get_zmm(k, 2), get_zmm(k, 3));
        vpunpckhbw(get_zmm(k, 1), get_zmm(k, 2), get_zmm(k, 3));

        vpunpcklwd(get_zmm(k, 2), get_zmm(k, 4), get_zmm(k, 0));
        vpunpckhwd(get_zmm(k, 3), get_zmm(k, 4), get_zmm(k, 0));
        vpunpcklwd(get_zmm(k, 4), get_zmm(k, 5), get_zmm(k, 1));
        vpunpckhwd(get_zmm(k, 5), get_zmm(k, 5), get_zmm(k, 1));

        vmovups(get_zmm(k, 0), vreg_idx_lo_256);
        vpermi2q(get_zmm(k, 0), get_zmm(k, 2), get_zmm(k, 4));
        vmovups(get_zmm(k, 1), vreg_idx_hi_256);
        vpermi2q(get_zmm(k, 1), get_zmm(k, 2), get_zmm(k, 4));
        vmovups(get_zmm(k, 2), vreg_idx_lo_256);
        vpermi2q(get_zmm(k, 2), get_zmm(k, 3), get_zmm(k, 5));
        vmovups(get_zmm(k, 4), vreg_idx_hi_256);
        vpermi2q(get_zmm(k, 4), get_zmm(k, 3), get_zmm(k, 5));

        vmovups(get_zmm(k, 3), vreg_idx_lo_128);
        vpermi2q(get_zmm(k, 3), get_zmm(k, 0), get_zmm(k, 2));
        dim_t tr_src_off_base = (kb * max_unroll + k) * tr_src_stride;
        vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base), get_zmm(k, 3));
        if (do_compute_compensation)
            vpdpbusd(get_comp_acc(0), zmm_comp_mul, get_zmm(k, 3));

        if (ncolumns > 16) {
            vmovups(get_zmm(k, 5), vreg_idx_hi_128);
            vpermi2q(get_zmm(k, 5), get_zmm(k, 0), get_zmm(k, 2));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    get_zmm(k, 5));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(1), zmm_comp_mul, get_zmm(k, 5));
        } else if (conf_->wei_n_blk > 16) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 64),
                    zmm_zero);
        }

        if (ncolumns > 32) {
            vmovups(get_zmm(k, 0), vreg_idx_lo_128);
            vpermi2q(get_zmm(k, 0), get_zmm(k, 1), get_zmm(k, 4));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    get_zmm(k, 0));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(2), zmm_comp_mul, get_zmm(k, 0));
        } else if (conf_->wei_n_blk > 32) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 128),
                    zmm_zero);
        }

        if (ncolumns > 48) {
            vmovups(get_zmm(k, 2), vreg_idx_hi_128);
            vpermi2q(get_zmm(k, 2), get_zmm(k, 1), get_zmm(k, 4));
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    get_zmm(k, 2));
            if (do_compute_compensation)
                vpdpbusd(get_comp_acc(3), zmm_comp_mul, get_zmm(k, 2));
        } else if (conf_->wei_n_blk > 48) {
            vmovups(EVEX_compress_addr(reg_tr_src, tr_src_off_base + 192),
                    zmm_zero);
        }
    }
}

void jit_brgemm_matmul_copy_b_int8_t::generate() {
    preamble();
    vpxord(zmm_zero, zmm_zero, zmm_zero);
    src_stride = conf_->N * typesize;
    tr_src_stride = conf_->LDB * k_blk_step * typesize;
    is_amx = conf_->isa == avx512_core_bf16_amx_int8;
    do_compute_compensation
            = conf_->s8s8_compensation_required || conf_->has_zero_point_a;

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);

    auto vmovdqa64 = [=](Zmm z, const void *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    alignas(64) static constexpr const int64_t idx_lo_256[8]
            = {0, 1, 2, 3, 8, 9, 10, 11};
    alignas(64) static constexpr const int64_t idx_hi_256[8]
            = {4, 5, 6, 7, 12, 13, 14, 15};

    alignas(64) static constexpr const int64_t idx_lo_128[8]
            = {0, 1, 8, 9, 4, 5, 12, 13};
    alignas(64) static constexpr const int64_t idx_hi_128[8]
            = {2, 3, 10, 11, 6, 7, 14, 15};
    alignas(64) static constexpr const uint8_t idx_lo_16[64]
            = {0, 1, 64, 65, 4, 5, 68, 69, 2, 3, 66, 67, 6, 7, 70, 71, 8, 9, 72,
                    73, 12, 13, 76, 77, 10, 11, 74, 75, 14, 15, 78, 79, 16, 17,
                    80, 81, 20, 21, 84, 85, 18, 19, 82, 83, 22, 23, 86, 87, 24,
                    25, 88, 89, 28, 29, 92, 93, 26, 27, 90, 91, 30, 31, 94, 95};

    alignas(64) static constexpr const uint8_t idx_hi_16[64] = {32, 33, 96, 97,
            36, 37, 100, 101, 34, 35, 98, 99, 38, 39, 102, 103, 40, 41, 104,
            105, 44, 45, 108, 109, 42, 43, 106, 107, 46, 47, 110, 111, 48, 49,
            112, 113, 52, 53, 116, 117, 50, 51, 114, 115, 54, 55, 118, 119, 56,
            57, 120, 121, 60, 61, 124, 125, 58, 59, 122, 123, 62, 63, 126, 127};

    alignas(64) static constexpr const uint8_t idx_lo_8[64]
            = {0, 64, 2, 66, 1, 65, 3, 67, 8, 72, 10, 74, 9, 73, 11, 75, 4, 68,
                    6, 70, 5, 69, 7, 71, 12, 76, 14, 78, 13, 77, 15, 79, 16, 80,
                    18, 82, 17, 81, 19, 83, 24, 88, 26, 90, 25, 89, 27, 91, 20,
                    84, 22, 86, 21, 85, 23, 87, 28, 92, 30, 94, 29, 93, 31, 95};

    alignas(64) static constexpr const uint8_t idx_hi_8[64] = {32, 96, 34, 98,
            33, 97, 35, 99, 40, 104, 42, 106, 41, 105, 43, 107, 36, 100, 38,
            102, 37, 101, 39, 103, 44, 108, 46, 110, 45, 109, 47, 111, 48, 112,
            50, 114, 49, 113, 51, 115, 56, 120, 58, 122, 57, 121, 59, 123, 52,
            116, 54, 118, 53, 117, 55, 119, 60, 124, 62, 126, 61, 125, 63, 127};

    vmovdqa64(vreg_idx_lo_256,
            is_amx ? (const void *)idx_lo_16 : (const void *)idx_lo_256);
    vmovdqa64(vreg_idx_hi_256,
            is_amx ? (const void *)idx_hi_16 : (const void *)idx_hi_256);
    vmovdqa64(vreg_idx_lo_128,
            is_amx ? (const void *)idx_lo_8 : (const void *)idx_lo_128);
    vmovdqa64(vreg_idx_hi_128,
            is_amx ? (const void *)idx_hi_8 : (const void *)idx_hi_128);

    if (do_compute_compensation) {
        int n_iters = div_up(conf_->wei_n_blk, 16);
        for (int i = 0; i < n_iters; i++)
            vpxord(get_comp_acc(i), get_comp_acc(i), get_comp_acc(i));
        mov(imm_addr64, 1);
        vpbroadcastb(zmm_comp_mul, imm_addr64.cvt8());
    }

    auto compute_K_loop = [=](bool is_N_tail) {
        const int k_unroll = 4;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_4x64_vnni(k_unroll * k_blk_step, ncolumns);
        add(reg_src, k_unroll * k_blk_step * src_stride);
        add(reg_tr_src, k_unroll * tr_src_stride);

        sub(reg_K_iters, k_unroll * k_blk_step);
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_4x64_vnni(k_blk_step, ncolumns);
        add(reg_src, k_blk_step * src_stride);
        add(reg_tr_src, tr_src_stride);

        sub(reg_K_iters, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_4x64_vnni(k_blk_tail, ncolumns);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_K_loop(false);
    L(done);

    if (do_compute_compensation) {
        const bool req_s8s8_comp = conf_->s8s8_compensation_required;
        const bool req_zp_comp = conf_->has_zero_point_a;
        int n_iters = div_up(conf_->wei_n_blk, 16);
        assert(IMPLICATION(req_zp_comp,
                conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

        // copy 'comp_acc' into s8s8_comp accumulator
        if (req_s8s8_comp) {
            for (int i = 0; i < n_iters; i++)
                vmovups(get_zmm_oscale_comp_res(i), get_comp_acc(i));
        }

        Label skip_acc, store;
        if (req_s8s8_comp)
            mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        if (req_zp_comp)
            mov(reg_zp_comp_ptr, ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        cmp(reg_K_start, 0);
        je(skip_acc, T_NEAR);
        if (req_s8s8_comp) {
            for (int i = 0; i < n_iters; i++) {
                const auto zmm_acc = get_comp_acc(i);
                const auto zmm_res = get_zmm_oscale_comp_res(i);
                const auto addr = EVEX_compress_addr(reg_comp_ptr, i * 64);
                vpaddd(zmm_res, zmm_acc, addr);
            }
        }

        if (req_zp_comp) {
            for (int i = 0; i < n_iters; i++) {
                const auto zmm_acc = get_comp_acc(i);
                const auto zmm_res = get_zmm_zp_comp_res(i);
                const auto addr = EVEX_compress_addr(reg_zp_comp_ptr, i * 64);
                vpaddd(zmm_res, zmm_acc, addr);
            }
        }

        L(skip_acc);
        cmp(reg_K_start, rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk);
        jl(store, T_NEAR);

        if (req_s8s8_comp) {
            mov(imm_addr64, 0xffffffff);
            const auto zmm_all_bits_1 = zmm_comp_mul;
            vpbroadcastd(zmm_all_bits_1, imm_addr64.cvt32());
            mov(imm_addr64, 0x1);
            const auto zmm_one_s32 = zmm_zero;
            vpbroadcastd(zmm_one_s32, imm_addr64.cvt32());

            for (int i = 0; i < n_iters; i++) {
                const auto zmm_res = get_zmm_oscale_comp_res(i);
                // multiply by 128
                vpslld(zmm_res, zmm_res, 7);
                // change sign
                vpandnq(zmm_res, zmm_res, zmm_all_bits_1);
                vpaddd(zmm_res, zmm_res, zmm_one_s32);
            }
        }

        if (req_zp_comp) {
            mov(reg_zp_a_neg_val_ptr,
                    ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
            const auto zmm_zp_a_neg_val = vreg_idx_hi_128;
            vbroadcastss(zmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);

            for (int i = 0; i < n_iters; i++) {
                const auto zmm_res = get_zmm_zp_comp_res(i);
                vpmulld(zmm_res, zmm_res, zmm_zp_a_neg_val);
            }
        }

        L(store);
        if (req_s8s8_comp) {
            for (int i = 0; i < n_iters; i++) {
                const auto zmm_res = get_zmm_oscale_comp_res(i);
                const auto addr = EVEX_compress_addr(reg_comp_ptr, i * 64);
                vmovups(addr, zmm_res);
            }
        }
        if (req_zp_comp) {
            for (int i = 0; i < n_iters; i++) {
                const auto zmm_res = get_zmm_zp_comp_res(i);
                const auto addr = EVEX_compress_addr(reg_zp_comp_ptr, i * 64);
                vmovups(addr, zmm_res);
            }
        }
    }

    postamble();
}

struct jit_brgemm_matmul_copy_b_bf16_t : public jit_brgemm_matmul_copy_b_t,
                                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_bf16_t)

    jit_brgemm_matmul_copy_b_bf16_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;
    using ymm = const Xbyak::Ymm;

    enum { typesize = sizeof(int16_t), k_blk_step = 2, n_blk_step = 16 };
    dim_t src_stride = 0, tr_src_stride = 0;

    opmask_t kTail = k7;
    opmask_t kFFFF = k6;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = r15;

    zmm zmm_permw = zmm30;
    zmm zmm_zero = zmm31;

    void copy_2x32_vnni(int nrows, int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_bf16_t::copy_2x32_vnni(int nrows, int ncolumns) {

    auto kmovd = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    };

    const int columns_tail = ncolumns % n_blk_step;
    const auto tail_mask = (1 << columns_tail) - 1;
    if (columns_tail < n_blk_step) kmovd(kTail, tail_mask);

    const int blk_sz = k_blk_step;
    const int max_regs_available = 30;
    const int max_unroll = max_regs_available / blk_sz;
    auto get_zmm = [=](int blk, int idx) {
        assert(idx >= 0 && idx < blk_sz && blk >= 0);
        auto reg_idx = max_unroll * ((idx + 1) % blk_sz) + blk;
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return zmm(reg_idx);
    };

    auto load = [=](int blk, int k, int n, opmask_t current_mask) {
        auto src_reg = get_zmm(blk, k % k_blk_step);
        auto src_load = src_reg | current_mask | T_z;
        vmovdqu16(src_load,
                EVEX_compress_addr(reg_src, k * src_stride + n * typesize));
    };

    int iter = 0;
    for_(int k = 0; k < nrows; k += k_blk_step)
    for (int n = 0; n < conf_->wei_n_blk; n += n_blk_step) {
        const int k_blk = k / k_blk_step;
        const dim_t tr_src_off
                = k_blk * tr_src_stride + n * k_blk_step * typesize;
        const auto store_addr = EVEX_compress_addr(reg_tr_src, tr_src_off);
        if (ncolumns - n <= 0) {
            vmovups(store_addr, zmm_zero);
            continue;
        }

        const opmask_t curr_msk = ncolumns - n < n_blk_step ? kTail : kFFFF;
        const int blk_idx = iter % max_unroll;
        load(blk_idx, k, n, curr_msk);
        const auto src_zmm0 = get_zmm(blk_idx, 0);
        if (nrows - k >= k_blk_step) {
            load(blk_idx, k + 1, n, curr_msk);
            const auto src_zmm1 = get_zmm(blk_idx, 1);
            const auto src_ymm1 = ymm(src_zmm1.getIdx());
            vinsertf64x4(src_zmm0, src_zmm0, src_ymm1, 1);
        }

        vpermw(src_zmm0, zmm_permw, src_zmm0);

        vmovups(store_addr, src_zmm0);
        iter++;
    }
}

void jit_brgemm_matmul_copy_b_bf16_t::generate() {
    preamble();
    vpxord(zmm_zero, zmm_zero, zmm_zero);
    src_stride = conf_->N * typesize;
    tr_src_stride = conf_->LDB * k_blk_step * typesize;

    alignas(64) static constexpr const int16_t bf16_vnni_permute[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(kFFFF, 0xffff); // 1111111111111111
    auto vmovdqa64 = [=](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator::vmovdqa64(z, ptr[imm_addr64]);
    };

    vmovdqa64(zmm_permw, (const int64_t *)bf16_vnni_permute);

    auto compute_K_loop = [=](bool is_N_tail) {
        const int k_unroll = 8;
        int ncolumns = is_N_tail ? conf_->N_tail : conf_->N_blk;

        Label K_loop_unrolled, K_loop_single, K_loop_tail_or_done;
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jl(K_loop_single, T_NEAR);

        L(K_loop_unrolled);
        copy_2x32_vnni(k_unroll * k_blk_step, ncolumns);
        add(reg_src, k_unroll * k_blk_step * src_stride);
        add(reg_tr_src, k_unroll * tr_src_stride);

        sub(reg_K_iters, k_unroll * k_blk_step);
        cmp(reg_K_iters, k_unroll * k_blk_step);
        jge(K_loop_unrolled, T_NEAR);

        L(K_loop_single);
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);

        copy_2x32_vnni(k_blk_step, ncolumns);
        add(reg_src, k_blk_step * src_stride);
        add(reg_tr_src, tr_src_stride);

        sub(reg_K_iters, k_blk_step);
        jmp(K_loop_single, T_NEAR);

        L(K_loop_tail_or_done);

        int k_blk_tail = conf_->K % k_blk_step;
        if (k_blk_tail > 0) {
            Label K_loop_done;
            cmp(reg_K_iters, 0);
            jle(K_loop_done, T_NEAR);

            copy_2x32_vnni(k_blk_tail, ncolumns);
            sub(reg_K_iters, k_blk_tail);
            L(K_loop_done);
        }
    };

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_K_loop(true);
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_K_loop(false);
    L(done);

    postamble();
}

struct jit_brgemm_matmul_copy_b_f32_t : public jit_brgemm_matmul_copy_b_t,
                                        public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_f32_t)

    jit_brgemm_matmul_copy_b_f32_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , src_stride_(conf_->N * typesize)
        , tr_src_stride_(conf_->LDB * typesize) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum { typesize = sizeof(float), n_blk_step = 16, max_regs_available = 30 };
    dim_t src_stride_, tr_src_stride_;

    opmask_t kTail = k7;
    opmask_t kFFFF = k6;

    reg64_t reg_src = rax;
    reg64_t reg_tr_src = rbx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_blk = r9;
    reg64_t reg_K_start = r10;
    reg32_t regw_tmp = r14d;
    reg64_t imm_addr64 = r15;

    zmm zmm_permw = zmm30;
    zmm zmm_zero = zmm31;

    inline void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovd(k, regw_tmp);
    }
    void copy_16_x_n_block(int nrows, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_f32_t::copy_16_x_n_block(
        int nrows, int ncolumns) {

    auto get_zmm = [=](int reg_idx) {
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return zmm(reg_idx);
    };

    auto load = [=](int blk, int k, int n, opmask_t current_mask) {
        auto src_zmm = get_zmm(blk);
        auto src_zmm_m = src_zmm | current_mask | T_z;
        vmovups(src_zmm_m,
                EVEX_compress_addr(reg_src, k * src_stride_ + n * typesize));
    };

    const int columns_tail = ncolumns % n_blk_step;
    const auto tail_mask = (1 << columns_tail) - 1;
    if (columns_tail < n_blk_step) kmovw(kTail, tail_mask);

    int iter = 0;
    for_(int k = 0; k < nrows; k++)
    for (int n = 0; n < conf_->wei_n_blk; n += n_blk_step) {
        const dim_t tr_src_off = k * tr_src_stride_ + n * typesize;
        const auto store_addr = EVEX_compress_addr(reg_tr_src, tr_src_off);

        const int zero_padding = ncolumns - n;
        if (zero_padding <= 0) {
            vmovups(store_addr, zmm_zero);
            continue;
        }

        const opmask_t curr_msk = zero_padding < n_blk_step ? kTail : kFFFF;
        const int blk_idx = iter % max_regs_available;
        load(blk_idx, k, n, curr_msk);

        const auto src_zmm0 = get_zmm(blk_idx);
        vmovups(store_addr, src_zmm0);
        iter++;
    }
}

void jit_brgemm_matmul_copy_b_f32_t::compute_k_loop(int ncolumns) {

    auto compute_uni_k_loop = [&](int unroll) {
        Label K_start_label, K_end_label;

        L(K_start_label);
        cmp(reg_K_iters, unroll);
        jl(K_end_label, T_NEAR);

        copy_16_x_n_block(unroll, ncolumns);
        add(reg_src, unroll * src_stride_);
        add(reg_tr_src, unroll * tr_src_stride_);

        sub(reg_K_iters, unroll);
        jmp(K_start_label, T_NEAR);

        L(K_end_label);
    };

    constexpr int k_unroll = 16;
    compute_uni_k_loop(k_unroll);
    compute_uni_k_loop(1);
}

void jit_brgemm_matmul_copy_b_f32_t::generate() {
    preamble();
    vpxord(zmm_zero, zmm_zero, zmm_zero);

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_blk, ptr[param1 + GET_OFF(current_N_blk)]);
    kmovw(kFFFF, 0xffff); // 1111111111111111

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp(reg_N_blk, conf_->N_tail);
        jne(not_N_tail, T_NEAR);
        compute_k_loop(conf_->N_tail);
        jmp(done, T_NEAR);

        L(not_N_tail);
    }

    compute_k_loop(conf_->N_blk);
    L(done);

    postamble();
}

struct jit_brgemm_matmul_copy_b_transposed_t
    : public jit_brgemm_matmul_copy_b_t,
      public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_transposed_t)

    jit_brgemm_matmul_copy_b_transposed_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , typesize(conf_->b_dt_sz)
        , vnni_granularity(granularity_max / typesize)
        , k_blk_step(bytes_in_zmm / typesize)
        , do_compute_compensation(
                  conf_->has_zero_point_a || conf_->s8s8_compensation_required)
        , req_zp_comp(conf_->has_zero_point_a)
        , req_s8s8_comp(conf_->s8s8_compensation_required) {}

    void operator()(ctx_t *ctx) override { jit_generator::operator()(ctx); }
    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    using opmask_t = const Xbyak::Opmask;
    using zmm = const Xbyak::Zmm;

    enum {
        granularity_max = 4,
        n_blk_step = 16,
        bytes_in_zmm = 64,
    };

    const int typesize;
    const int vnni_granularity;
    const int k_blk_step;
    const bool do_compute_compensation;
    const bool req_zp_comp;
    const bool req_s8s8_comp;

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
    reg64_t reg_comp_ptr = rdx;

    reg64_t reg_K_iters = r8;
    reg64_t reg_N_iters = r9;
    reg64_t reg_src = r10;
    reg64_t reg_tr_src = r11;
    reg64_t reg_zp_comp_ptr = r12;
    reg64_t reg_zp_a_neg_val_ptr = r13;
    reg64_t reg_K_start = r14;

    reg64_t regq_tmp = r15;
    reg32_t regw_tmp = r15d;
    reg64_t imm_addr64 = abi_not_param1;

    zmm zmm_zp_a_neg_val = zmm29;
    zmm zmm_comp_acc = zmm30;
    zmm zmm_comp_mul = zmm31;
    zmm zmm_s8s8_comp_acc = zmm28;
    zmm zmm_all_bits_1 = zmm27;
    zmm zmm_one_s32 = zmm26;

    void copy_16x64_vnni(int nrows, int ncolumns);
    void compute_K_loop(bool is_N_tail, int curr_K_tail, bool is_first_K_iter,
            bool is_last_K_iter);
    void compute_N_loop(
            int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter);

    void generate() override;
};

void jit_brgemm_matmul_copy_b_transposed_t::copy_16x64_vnni(
        int nrows, int ncolumns) {
    assert(nrows >= 0 && nrows <= n_blk_step && ncolumns >= 0
            && ncolumns <= k_blk_step);
    if (!nrows) return;

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        // If compensation compute is required - last 6 zmms are reserved for it
        assert(i >= 0 && i < 16 - do_compute_compensation * 6);
        return Zmm(16 + i);
    };

    auto kmovq = [=](Opmask k, size_t q) {
        mov(regq_tmp, q);
        jit_generator::kmovq(k, regq_tmp);
    };

    const int columns_tail = ncolumns % k_blk_step;
    const auto tail_mask = size_t(((size_t)1 << (typesize * columns_tail)) - 1);
    if (columns_tail > 0) kmovq(kTail, tail_mask);

    auto load = [=](int i) {
        auto src_reg = src_zmm(i);
        if (i >= nrows) {
            vpxord(src_reg, src_reg, src_reg);
            return;
        }

        auto src_load = columns_tail > 0 ? src_reg | kTail | T_z : src_reg;
        vmovdqu8(src_load, EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride);
        vmovups(addr, r);
    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);
        // If compensation compute is required - use tmp(0) ... tmp(7)
        // to not spoil reserved registers' values
        const int tmp_corr_idx = do_compute_compensation * base_idx;

        // swap 1
        for (int i = 0; i < 4; i++) {
            const int src_idx0 = base_idx + i * 2;
            const int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                load(src_idx1);
            }

            const auto tmp0 = tmp_zmm(src_idx0 - tmp_corr_idx);
            const auto tmp1 = tmp_zmm(src_idx1 - tmp_corr_idx);
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

            const auto tmp0 = tmp_zmm(src_idx0 - tmp_corr_idx);
            const auto tmp1 = tmp_zmm(src_idx2 - tmp_corr_idx);
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

            const auto tmp0 = tmp_zmm(src_idx0 - tmp_corr_idx);
            const auto src0 = src_zmm(src_idx0);
            const auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
        }
    };

    auto fixup16x16 = [=]() {
        // swap 8
        for (int i = 0; i < 8; i++) {
            const auto tmp = tmp_zmm(i);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            if (do_compute_compensation)
                vpdpbusd(zmm_comp_acc, zmm_comp_mul, tmp);
            store(tmp, i);
        }

        for (int i = 0; i < 8; i++) {
            // If compensation compute is required - last 4 zmms are reserved
            const auto tmp = IMPLICATION(do_compute_compensation, i < 2)
                    ? tmp_zmm(8 + i)
                    : src_zmm((i - 2) / 2 + (i % 2) * 8);
            const auto src0 = src_zmm(i);
            const auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            if (do_compute_compensation)
                vpdpbusd(zmm_comp_acc, zmm_comp_mul, tmp);
            store(tmp, 8 + i);
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();
}

void jit_brgemm_matmul_copy_b_transposed_t::compute_K_loop(bool is_N_tail,
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    MAYBE_UNUSED(is_first_K_iter);
    MAYBE_UNUSED(is_last_K_iter);
    const int N_chunk_tail = conf_->N % n_blk_step;
    int nrows = is_N_tail ? N_chunk_tail : n_blk_step;
    if (do_compute_compensation)
        vpxord(zmm_comp_acc, zmm_comp_acc, zmm_comp_acc);

    Label K_loop, K_loop_tail_or_done;
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);

    mov(reg_src, reg_src_base);
    mov(reg_tr_src, reg_tr_src_base);
    if (curr_K_tail > 0) {
        cmp(reg_K_iters, k_blk_step);
        jl(K_loop_tail_or_done, T_NEAR);
    }

    L(K_loop);
    copy_16x64_vnni(nrows, k_blk_step);
    add(reg_src, k_blk_step * typesize);
    add(reg_tr_src, k_blk_step / vnni_granularity * tr_src_stride);

    sub(reg_K_iters, k_blk_step);
    cmp(reg_K_iters, k_blk_step);
    jge(K_loop, T_NEAR);

    L(K_loop_tail_or_done);

    if (curr_K_tail > 0) copy_16x64_vnni(nrows, curr_K_tail);

    if (req_s8s8_comp) {
        const auto addr = zword[reg_comp_ptr];
        if (!is_first_K_iter)
            vpaddd(zmm_s8s8_comp_acc, zmm_comp_acc, addr);
        else
            vmovups(zmm_s8s8_comp_acc, zmm_comp_acc);

        if (is_last_K_iter) {
            // multiply by 128
            vpslld(zmm_s8s8_comp_acc, zmm_s8s8_comp_acc, 7);
            // change sign
            vpandnq(zmm_s8s8_comp_acc, zmm_s8s8_comp_acc, zmm_all_bits_1);
            vpaddd(zmm_s8s8_comp_acc, zmm_s8s8_comp_acc, zmm_one_s32);
        }
        vmovups(addr, zmm_s8s8_comp_acc);
    }
    if (req_zp_comp) {
        const auto addr = zword[reg_zp_comp_ptr];
        if (!is_first_K_iter) vpaddd(zmm_comp_acc, zmm_comp_acc, addr);
        if (is_last_K_iter)
            vpmulld(zmm_comp_acc, zmm_comp_acc, zmm_zp_a_neg_val);
        vmovups(addr, zmm_comp_acc);
    }
}

void jit_brgemm_matmul_copy_b_transposed_t::compute_N_loop(
        int curr_K_tail, bool is_first_K_iter, bool is_last_K_iter) {
    const int N_chunk_tail = conf_->N % n_blk_step;
    const size_t comp_shift = 64;

    Label N_loop, N_loop_tail_or_done;
    if (N_chunk_tail > 0) {
        cmp(reg_N_iters, n_blk_step);
        jl(N_loop_tail_or_done, T_NEAR);
    }

    L(N_loop);
    compute_K_loop(false, curr_K_tail, is_first_K_iter, is_last_K_iter);
    add(reg_src_base, n_blk_step * src_stride);
    add(reg_tr_src_base, n_blk_step * vnni_granularity * typesize);
    if (req_zp_comp) add(reg_zp_comp_ptr, comp_shift);
    if (req_s8s8_comp) add(reg_comp_ptr, comp_shift);

    sub(reg_N_iters, n_blk_step);
    cmp(reg_N_iters, n_blk_step);
    jge(N_loop, T_NEAR);

    L(N_loop_tail_or_done);
    if (N_chunk_tail > 0) {
        Label N_loop_done;
        cmp(reg_N_iters, 0);
        jle(N_loop_done, T_NEAR);

        compute_K_loop(true, curr_K_tail, is_first_K_iter, is_last_K_iter);
        L(N_loop_done);
    }
}

void jit_brgemm_matmul_copy_b_transposed_t::generate() {

    preamble();

    src_stride = conf_->K * typesize;
    tr_src_stride = conf_->LDB * vnni_granularity * typesize;

    mov(reg_src_base, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src_base, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_K_iters, ptr[param1 + GET_OFF(current_K_iters)]);
    mov(reg_N_iters, ptr[param1 + GET_OFF(current_N_blk)]);

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

    const dim_t N_chunk_elems = conf_->N_chunk_elems;
    assert(N_chunk_elems % n_blk_step == 0 || N_chunk_elems == conf_->N);
    UNUSED(N_chunk_elems);

    const auto K_blk_tail = nstl::min(conf_->K, conf_->K_blk) % k_blk_step;
    const auto K_tail_tail = (conf_->K % conf_->K_blk) % k_blk_step;

    auto compute_body = [=](bool is_first_K_iter, bool is_last_K_iter) {
        if (is_last_K_iter) {
            if (req_s8s8_comp) {
                mov(imm_addr64, 0xffffffff);
                vpbroadcastd(zmm_all_bits_1, imm_addr64.cvt32());
                mov(imm_addr64, 0x1);
                vpbroadcastd(zmm_one_s32, imm_addr64.cvt32());
            }
            if (req_zp_comp) {
                mov(reg_zp_a_neg_val_ptr,
                        ptr[param1 + GET_OFF(zp_a_neg_value_ptr)]);
                vbroadcastss(zmm_zp_a_neg_val, ptr[reg_zp_a_neg_val_ptr]);
            }
        }

        Label compute_body_done;
        if (conf_->K_tail > 0 && K_blk_tail != K_tail_tail) {
            Label not_K_tail;
            cmp(reg_K_iters, conf_->K_blk);
            je(not_K_tail, T_NEAR);
            compute_N_loop(K_tail_tail, is_first_K_iter, is_last_K_iter);
            jmp(compute_body_done, T_NEAR);

            L(not_K_tail);
        }

        compute_N_loop(K_blk_tail, is_first_K_iter, is_last_K_iter);
        L(compute_body_done);
    };

    Label done;
    if (do_compute_compensation) {
        assert(IMPLICATION(req_zp_comp,
                conf_->src_zp_type == brgemm_broadcast_t::per_tensor));

        mov(reg_K_start, ptr[param1 + GET_OFF(current_K_start)]);
        if (req_s8s8_comp)
            mov(reg_comp_ptr, ptr[param1 + GET_OFF(compensation_ptr)]);
        if (req_zp_comp)
            mov(reg_zp_comp_ptr, ptr[param1 + GET_OFF(zp_a_compensation_ptr)]);

        mov(regq_tmp, 1);
        vpbroadcastb(zmm_comp_mul, regq_tmp.cvt8());

        const auto last_K_threshold
                = rnd_up(conf_->K, conf_->K_blk) - conf_->K_blk;
        Label not_first, not_first_not_last;
        cmp(reg_K_start, 0);
        jne(not_first, T_NEAR);
        {
            // first K iteration
            Label first_not_last;
            cmp(reg_K_start, last_K_threshold);
            jl(first_not_last, T_NEAR);
            compute_body(true, true);
            jmp(done, T_NEAR);

            L(first_not_last);
            compute_body(true, false);
            jmp(done, T_NEAR);
        }

        L(not_first);
        cmp(reg_K_start, last_K_threshold);
        jl(not_first_not_last, T_NEAR);

        compute_body(false, true);
        jmp(done, T_NEAR);
        L(not_first_not_last);
    }

    compute_body(false, false);
    L(done);

    postamble();
}

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    const bool is_B_transposed
            = one_of(conf->wei_tag, ba, acb, abdc, abced, abcdfe, abcdegf,
                    abcdefhg, abcdefgih, abcdefghji, abcdefghikj, abcdefghijlk);
    const bool is_bf16
            = everyone_is(data_type::bf16, conf->src_dt, conf->wei_dt);
    const bool is_f32 = everyone_is(data_type::f32, conf->src_dt, conf->wei_dt);
    if (is_B_transposed) {
        CHECK(safe_ptr_assign(
                copy_ker, new jit_brgemm_matmul_copy_b_transposed_t(conf)));
    } else {
        if (is_bf16) {
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_b_bf16_t(conf)));
        } else if (is_f32) {
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_b_f32_t(conf)));
        } else {
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_b_int8_t(conf)));
        }
    }

    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    if (conf->transposed_A) {
        CHECK(safe_ptr_assign(copy_ker,
                new jit_brgemm_matmul_copy_a_transposed_impl_t(conf)));
    } else {
        CHECK(safe_ptr_assign(
                copy_ker, new jit_brgemm_matmul_copy_a_impl_t(conf)));
    }

    return copy_ker->create_kernel();
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
