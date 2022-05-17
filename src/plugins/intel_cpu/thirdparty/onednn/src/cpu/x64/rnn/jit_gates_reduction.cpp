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

#include "cpu/x64/rnn/jit_gates_reduction.hpp"

#include <cmath>
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_gates_reduction_t::jit_gates_reduction_t(
        const rnn_utils::rnn_conf_t &rnn, bool is_n_tail)
    : rnn_(rnn)
    , is_n_tail_(is_n_tail)
    , n_block_(is_n_tail_ ? rnn_.diff_wei_brgemm.n_tail
                          : rnn_.diff_wei_brgemm.n_block)
    , n_simd_w_blks_(n_block_ / simd_w_)
    , n_tail_(n_block_ % simd_w_)
    , bf16_ones_(rnn_.is_bf16() ? reserve_vmm() : 0)
    , acc_regs_(reserve_acc_regs()) {}

void jit_gates_reduction_t::generate() {
    preamble();
    load_addresses();
    init();
    compute_loop();
    store_data();
    postamble();
}

#define PARAM_OFF(x) offsetof(jit_gates_reduction_t::call_params_t, x)

size_t jit_gates_reduction_t::reserve_vmm() {
    return number_reserved_vmms_++;
}

std::vector<Xbyak::Zmm> jit_gates_reduction_t::reserve_acc_regs() {
    std::vector<Xbyak::Zmm> acc_regs;
    acc_regs.reserve(n_simd_w_blks_ + n_tail_);

    for (int i = 0; i < n_simd_w_blks_; ++i)
        acc_regs.emplace_back(Xbyak::Zmm(reserve_vmm()));

    if (n_tail_) acc_regs.emplace_back(Xbyak::Zmm(reserve_vmm()));

    return acc_regs;
}

void jit_gates_reduction_t::load_addresses() {
    mov(reg_src_, ptr[abi_param1 + PARAM_OFF(src)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst)]);
}

void jit_gates_reduction_t::init() {
    static constexpr auto off_step = simd_w_ * sizeof(float);

    for (int i = 0; i < n_simd_w_blks_; ++i)
        uni_vmovups(acc_regs_[i], ptr[reg_dst_ + (i * off_step)]);

    if (n_tail_) {
        const int mask_f32 = (1 << n_tail_) - 1;
        const Xbyak::Reg32 regw_tmp = reg_tmp_.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(tail_mask_, regw_tmp);

        uni_vmovups(acc_regs_.back() | tail_mask_ | T_z,
                ptr[reg_dst_ + (n_simd_w_blks_ * off_step)]);
    }

    if (rnn_.is_bf16()) {
        xor_(reg_tmp_, reg_tmp_);
        mov(reg_tmp_.cvt16(), bfloat16_t(1.0f).raw_bits_);
        const Xbyak::Xmm xmm_tmp(bf16_ones_.getIdx());
        vmovd(xmm_tmp, reg_tmp_.cvt32());
        vpbroadcastw(bf16_ones_, xmm_tmp);
    }
}

void jit_gates_reduction_t::compute_step(
        const Xbyak::Zmm &acc, const Xbyak::Address &addr, bool tail) {

    const auto dst = tail ? (acc | tail_mask_) : acc;

    if (rnn_.is_bf16())
        vdpbf16ps(dst, bf16_ones_, addr);
    else
        uni_vaddps(dst, acc, addr);
}

void jit_gates_reduction_t::compute(dim_t unrolling) {

    const int n_block_off = rnn_.diff_wei_brgemm.n_block * sizeof(float);

    for (dim_t k = 0; k < unrolling; ++k) {
        const int k_offset = -1 * (k + 1) * n_block_off;
        const int first_reversed_block = acc_regs_.size() - 1;

        for (int n_block = first_reversed_block; n_block >= 0; --n_block) {
            const bool tail = static_cast<bool>(n_tail_)
                    && n_block == first_reversed_block;
            const auto &acc_zmm = acc_regs_[n_block];
            const int nk_offset = k_offset + n_block * simd_w_ * sizeof(float);
            compute_step(acc_zmm, ptr[reg_src_ + reg_loop_ + nk_offset], tail);
        }
    }
}

void jit_gates_reduction_t::compute_loop() {
    const dim_t k_block = 32;
    const dim_t k_pack = rnn_.is_bf16() ? 2 : 1;
    const dim_t k = rnn_.diff_wei_brgemm.Kpadded;
    const auto res = std::div(k, k_block);
    const int n_block_off = rnn_.diff_wei_brgemm.n_block
            * (rnn_.is_bf16() ? sizeof(bfloat16_t) : sizeof(float));
    const auto &num_k_blks = res.quot;
    const auto &k_tail = res.rem;

    Xbyak::Label unroll_loop, unroll_loop_tail, end;

    mov(reg_loop_, k * n_block_off);

    const dim_t tail_bytes = k_tail * n_block_off;
    const dim_t block_bytes = k_block * n_block_off;

    L(unroll_loop);
    {
        if (num_k_blks) {
            cmp(reg_loop_, tail_bytes);
            jle(unroll_loop_tail, T_NEAR);
            compute(k_block / k_pack);

            sub(reg_loop_, block_bytes);
            jmp(unroll_loop);
        }
    }

    L(unroll_loop_tail);
    {
        if (tail_bytes) { compute(res.rem / k_pack); }
    }

    L(end);
}

void jit_gates_reduction_t::store_data() {
    static constexpr auto off_step = simd_w_ * sizeof(float);

    for (int i = 0; i < n_simd_w_blks_; ++i)
        uni_vmovups(ptr[reg_dst_ + (i * off_step)], acc_regs_[i]);

    if (n_tail_)
        uni_vmovups(ptr[reg_dst_ + (n_simd_w_blks_ * off_step)] | tail_mask_,
                acc_regs_.back());
}

#undef PARAM_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
