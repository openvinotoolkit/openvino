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

#include "cpu/x64/prelu/jit_prelu_reduction_kernel.hpp"
#include "common/nstl.hpp"
#include "cpu/x64/prelu/jit_prelu_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

static constexpr dim_t alignment
        = platform::get_cache_line_size() / sizeof(float);
static dim_t get_C(const cpu_prelu_bwd_pd_t *pd) {
    const memory_desc_wrapper src_diff_d {pd->diff_src_md(0)};
    return src_diff_d.ndims() >= 2 ? src_diff_d.dims()[1] : 1;
}

jit_prelu_reduction_kernel_t::jit_prelu_reduction_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, int simd_w)
    : scratchpad_c_block_offset_(
            utils::rnd_up(get_C(pd), alignment) * sizeof(float))
    , simd_w_(simd_w)
    , data_type_(pd->diff_weights_md(0)->data_type)
    , tail_size_(get_C(pd) % simd_w)
    , tail_block_size_(prelu::get_block_tail_size(pd->diff_weights_md(0)))
    , c_blk_nelems_(prelu::c_blk_nelems(pd->diff_weights_md(0), false)) {}

#define PARAM_OFF(x) offsetof(call_params_t, x)

size_t jit_prelu_reduction_kernel_t::simd_w() const {
    return simd_w_;
}

void jit_prelu_reduction_kernel_t::load_kernel_call_params() {
    mov(reg_reduction_blocks_, ptr[abi_param1 + PARAM_OFF(reduction_blocks)]);
    mov(reg_weights_diff_scratch_,
            ptr[abi_param1 + PARAM_OFF(weights_diff_scratch)]);
    mov(reg_weights_diff_, ptr[abi_param1 + PARAM_OFF(weights_diff)]);
    mov(reg_tail_, byte[abi_param1 + PARAM_OFF(tail)]);
    mov(reg_last_c_blk_byte_, byte[abi_param1 + PARAM_OFF(is_last_c_blk)]);
}

#undef PARAM_OFF

void jit_prelu_reduction_kernel_t::generate() {
    Xbyak::Label tail, end;

    preamble();
    load_kernel_call_params();

    if (tail_size_) {
        cmp(reg_tail_, 1);
        je(tail, T_NEAR);

        generate(false /* tail*/);
        jmp(end, T_NEAR);

        L(tail);
        generate(true /* tail*/);

        L(end);
    } else
        generate(false /* tail*/);

    postamble();
}

void jit_prelu_reduction_kernel_t::generate(bool tail) {

    Xbyak::Label unroll_loop, unroll_loop_tail, end;
    const auto unrolling_factor = get_unrolling_factor(tail);

    prepare_kernel_const_vars(tail);
    xor_(reg_offset_, reg_offset_);
    L(unroll_loop);
    {
        const size_t offt = unrolling_factor * scratchpad_c_block_offset_;
        cmp(reg_reduction_blocks_, unrolling_factor);
        jl(unroll_loop_tail, T_NEAR);
        compute_dst(unrolling_factor, tail);
        sub(reg_reduction_blocks_, unrolling_factor);
        add(reg_offset_, offt);
        jmp(unroll_loop);
    }

    L(unroll_loop_tail);
    {
        cmp(reg_reduction_blocks_, 0);
        jle(end, T_NEAR);
        compute_dst(1, tail);
        sub(reg_reduction_blocks_, 1);
        add(reg_offset_, scratchpad_c_block_offset_);
        jmp(unroll_loop_tail);
    }

    L(end);

    finalize(tail);
}

int jit_prelu_reduction_kernel_t::reserve_vmm() {
    return number_reserved_vmms_++;
}

Xbyak::Address jit_prelu_reduction_kernel_t::diff_scratch_ptr(
        int unrolling_group) const {
    return ptr[reg_weights_diff_scratch_ + reg_offset_
            + unrolling_group * scratchpad_c_block_offset_];
}

template <typename Vmm>
jit_uni_prelu_reduction_kernel_t<Vmm>::jit_uni_prelu_reduction_kernel_t(
        const cpu_prelu_bwd_pd_t *pd, const cpu_isa_t &isa)
    : jit_prelu_reduction_kernel_t(
            pd, prelu::vmm_traits_t<Vmm>::vlen / sizeof(float))
    , isa_(isa)
    , saturation_needed_(utils::one_of(
              data_type_, data_type::s8, data_type::u8, data_type::s32))
    , accumulator_(reserve_vmm())
    , tail_vmm_mask_(
              tail_size_ && utils::one_of(isa, avx, avx2) ? reserve_vmm() : 0)
    , saturation_lower_bound_(saturation_needed_ ? reserve_vmm() : 0)
    , saturation_upper_bound_(saturation_needed_ ? reserve_vmm() : 0)
    , io_(this, isa_, data_type_, {},
              io::io_tail_conf_t {simd_w_, tail_size_, tail_opmask_,
                      tail_vmm_mask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {},
              io::io_saturation_conf_t {saturation_lower_bound_.getIdx(),
                      saturation_upper_bound_.getIdx(), reg_tmp_}) {}

template <typename Vmm>
size_t jit_uni_prelu_reduction_kernel_t<Vmm>::get_unrolling_factor(
        bool tail) const {
    const size_t max_num_threads = dnnl_get_max_threads();
    const size_t n_vregs = prelu::get_n_vregs(isa_);
    const size_t number_of_available_regs = n_vregs
            - (number_reserved_vmms_
                    + (data_type_ == data_type::bf16 && isa_ == avx512_core
                                    ? 4
                                    : 0));

    return nstl::min(number_of_available_regs, max_num_threads);
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::finalize(bool tail) {
    io_.store(accumulator_, ptr[reg_weights_diff_], tail);

    if (!tail_block_size_) return;
    Xbyak::Label end;
    cmp(reg_last_c_blk_byte_, 1);
    jne(end, T_NEAR);
    const auto base_off = (c_blk_nelems_ % simd_w_) ? tail_size_ : simd_w_;
    prelu::apply_zero_padding(this, base_off, data_type_, tail_block_size_,
            reg_weights_diff_, nullptr);
    L(end);
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::prepare_kernel_const_vars(
        bool tail) {
    uni_vxorps(accumulator_, accumulator_, accumulator_);

    io_.init_bf16();
    if (tail) io_.prepare_tail_mask();
    if (saturation_needed_) io_.init_saturate_f32();
}

template <typename Vmm>
void jit_uni_prelu_reduction_kernel_t<Vmm>::compute_dst(
        int unrolling_factor, bool tail) {

    const int vmm_begin = number_reserved_vmms_;

    for (int unrolling_group = 0; unrolling_group < unrolling_factor;
            ++unrolling_group) {
        const Vmm load_vmm {vmm_begin + unrolling_group};
        uni_vmovups(load_vmm, diff_scratch_ptr(unrolling_group));
        uni_vaddps(accumulator_, accumulator_, load_vmm);
    }
}

jit_prelu_reduction_kernel_t *jit_prelu_reduction_kernel_t::create(
        const cpu_prelu_bwd_pd_t *pd) {

    const auto isa = prelu::get_supported_isa();

    if (is_superset(isa, avx512_common))
        return new jit_uni_prelu_reduction_kernel_t<Xbyak::Zmm>(pd, isa);
    else if (is_superset(isa, avx))
        if (isa == avx && prelu::is_s8u8({pd->diff_weights_md(0)->data_type}))
            return new jit_uni_prelu_reduction_kernel_t<Xbyak::Xmm>(pd, isa);
        else
            return new jit_uni_prelu_reduction_kernel_t<Xbyak::Ymm>(pd, isa);
    else if (isa == sse41)
        return new jit_uni_prelu_reduction_kernel_t<Xbyak::Xmm>(pd, isa);

    return nullptr;
}

template class jit_uni_prelu_reduction_kernel_t<Xbyak::Zmm>;
template class jit_uni_prelu_reduction_kernel_t<Xbyak::Ymm>;
template class jit_uni_prelu_reduction_kernel_t<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
