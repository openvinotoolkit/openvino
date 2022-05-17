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

#include "cpu/x64/rnn/jit_brgemm_transpose_single_row.hpp"

#include <cmath>
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_brgemm_transpose_single_row_t::jit_brgemm_transpose_single_row_t(
        const int m_block)
    : m_block_(m_block)
    , full_loop_iters_(m_block_ / (vmms_available_ * simd_w_))
    , tail_(m_block_ % simd_w_)
    , k_blocks_nb_(m_block_ / simd_w_) {}

void jit_brgemm_transpose_single_row_t::generate() {
    preamble();
    load_addresses();
    compute_loop();
    postamble();
}

#define PARAM_OFF(x) \
    offsetof(jit_brgemm_transpose_single_row_t::call_params_t, x)
void jit_brgemm_transpose_single_row_t::load_addresses() {
    mov(reg_src_, ptr[abi_param1 + PARAM_OFF(src)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst)]);
}
#undef PARAM

void jit_brgemm_transpose_single_row_t::compute(
        const dim_t unrolling, const bool is_tail) {

    if (is_tail) {
        mov(reg_tmp_.cvt32(), std::pow(2, tail_) - 1);
        kmovd(tail_mask_, reg_tmp_.cvt32());
    }

    for (int k = unrolling - 1; k >= 0; k--) {
        const auto read_vmm
                = is_tail ? Xbyak::Zmm(k) | tail_mask_ | T_z : Xbyak::Zmm(k);
        const auto src_off = k * simd_w_ * sizeof(bfloat16_t);
        vpmovzxwd(read_vmm, ptr[reg_src_ + src_off]);
    }

    for (int k = unrolling - 1; k >= 0; k--) {
        const auto store_vmm
                = is_tail ? Xbyak::Zmm(k) | tail_mask_ : Xbyak::Zmm(k);
        const auto dst_off = k * simd_w_ * sizeof(float);
        uni_vmovups(ptr[reg_dst_ + dst_off], store_vmm);
    }
}

void jit_brgemm_transpose_single_row_t::compute_loop() {
    Xbyak::Label unroll_full_loop, loop_end;

    if (full_loop_iters_ > 0) {
        const auto loop_l_off = vmms_available_ * simd_w_;
        const auto loop_src_off = loop_l_off * sizeof(bfloat16_t);
        const auto loop_dst_off = loop_l_off * sizeof(float);

        mov(reg_full_loop_, full_loop_iters_);
        L(unroll_full_loop);
        {
            cmp(reg_full_loop_, 0);
            je(loop_end, T_NEAR);

            compute(vmms_available_, false);

            add(reg_src_, loop_src_off);
            add(reg_dst_, loop_dst_off);

            dec(reg_full_loop_);
            jmp(unroll_full_loop);
        }
        L(loop_end);
    }

    const int k_blocks_left = k_blocks_nb_ - full_loop_iters_ * vmms_available_;
    if (k_blocks_left > 0) {
        const auto off = k_blocks_left * simd_w_;
        const auto src_off = off * sizeof(bfloat16_t);
        const auto dst_off = off * sizeof(float);

        compute(k_blocks_left, false);
        add(reg_src_, src_off);
        add(reg_dst_, dst_off);
    }

    if (tail_ > 0) compute(1, true);
}

#undef PARAM_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
