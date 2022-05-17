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

#include "cpu/x64/rnn/jit_diff_weights_peephole.hpp"
#include "common/c_types_map.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_diff_weights_peephole_t::jit_diff_weights_peephole_t(
        const rnn_utils::rnn_conf_t &rnn, const dim_t dhc_block_size)
    : c_states_dt_(rnn.src_iter_c_dt)
    , scratch_dt_(rnn.is_bf16() ? data_type::bf16 : data_type::f32)
    , dst_dt_(data_type::f32)
    , compute_block_size_(dhc_block_size)
    , tail_size_(dhc_block_size % simd_w_)
    , io_(this, mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core,
              {c_states_dt_, scratch_dt_, dst_dt_}, {},
              io::io_tail_conf_t {static_cast<std::size_t>(simd_w_),
                      static_cast<std::size_t>(tail_size_), tail_opmask_, 0,
                      reg_tmp_}) {}

void jit_diff_weights_peephole_t::generate() {
    preamble();
    load_addresses();
    init();
    compute_loop();
    postamble();
}

#define PARAM_OFF(x) offsetof(jit_diff_weights_peephole_t::call_params_t, x)

void jit_diff_weights_peephole_t::load_addresses() {
    mov(reg_c_states_, ptr[abi_param1 + PARAM_OFF(c_states)]);
    mov(reg_scratch_gates_, ptr[abi_param1 + PARAM_OFF(scratch_gates)]);
    mov(reg_dst_, ptr[abi_param1 + PARAM_OFF(dst)]);
}

#undef PARAM_OFF

void jit_diff_weights_peephole_t::init() {
    if (tail_size_) { io_.prepare_tail_mask(); }
}

void jit_diff_weights_peephole_t::compute_loop() {

    Xbyak::Label unroll_loop, unroll_loop_tail;

    mov(loop_cnt_, compute_block_size_);
    xor_(reg_offset_, reg_offset_);

    const size_t offt_max = max_unrolling * simd_w_;
    const size_t full_unroling_steps = compute_block_size_ / offt_max;

    if (full_unroling_steps) {
        L(unroll_loop);
        {
            cmp(loop_cnt_, offt_max);
            jl(unroll_loop_tail, T_NEAR);

            compute_dst(max_unrolling, false /*tail*/);
            sub(loop_cnt_, offt_max);
            add(reg_offset_, offt_max);
            jmp(unroll_loop);
        }
    }

    const size_t full_blocks_left = (compute_block_size_ - tail_size_
                                            - (full_unroling_steps * offt_max))
            / simd_w_;

    L(unroll_loop_tail);
    {
        if (full_blocks_left) {
            compute_dst(full_blocks_left, false /*tail*/);
            if (tail_size_) {
                const size_t offt = full_blocks_left * simd_w_;
                add(reg_offset_, offt);
            }
        }
        if (tail_size_) { compute_dst(1u /*unrolling factor*/, true /*tail*/); }
    }
}

void jit_diff_weights_peephole_t::compute_dst(
        size_t unrolling_factor, bool tail) {

    static constexpr dim_t number_vmm_single_compute = 3;

    const auto get_compute_zmm = [=](size_t base_idx, size_t unroll_group) {
        return Xbyak::Zmm(base_idx + unroll_group * number_vmm_single_compute);
    };

    const auto get_addr = [&](const Xbyak::Reg64 &reg_base, const dim_t offt,
                                  const data_type_t dt) {
        const auto dt_size = types::data_type_size(dt);
        return ptr[reg_base + reg_offset_ * dt_size + offt * dt_size];
    };

    static constexpr size_t dst_idx = 0;
    static constexpr size_t scratch_idx = 1;
    static constexpr size_t c_states_idx = 2;

    const auto io_dst = io_.at(dst_dt_);
    const auto io_scratch = io_.at(scratch_dt_);
    const auto io_c_states = io_.at(c_states_dt_);

    for (size_t unroll_group = 0; unroll_group < unrolling_factor;
            ++unroll_group) {

        const auto dst_zmm = get_compute_zmm(dst_idx, unroll_group);
        const auto scratch_zmm = get_compute_zmm(scratch_idx, unroll_group);
        const auto c_states_zmm = get_compute_zmm(c_states_idx, unroll_group);

        const auto unroll_offset = unroll_group * simd_w_;
        const auto dst_addr = get_addr(reg_dst_, unroll_offset, dst_dt_);
        io_dst->load(dst_addr, dst_zmm, tail);
        io_scratch->load(
                get_addr(reg_scratch_gates_, unroll_offset, scratch_dt_),
                scratch_zmm, tail);
        io_c_states->load(get_addr(reg_c_states_, unroll_offset, c_states_dt_),
                c_states_zmm, tail);
        const auto dst_zmm_masked = tail ? dst_zmm | tail_opmask_ : dst_zmm;
        uni_vfmadd231ps(dst_zmm_masked, scratch_zmm, c_states_zmm);
        io_dst->store(dst_zmm, dst_addr, tail);
    }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
