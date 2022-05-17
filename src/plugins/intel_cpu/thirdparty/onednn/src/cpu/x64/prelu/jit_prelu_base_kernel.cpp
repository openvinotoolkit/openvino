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

#include "cpu/x64/prelu/jit_prelu_base_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_prelu_base_kernel_t::jit_prelu_base_kernel_t(const cpu_isa_t &isa, int vlen,
        const prelu::bcast &bcast, const memory_desc_wrapper &tensor_md,
        size_t number_vmm_single_compute)
    : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
    , isa_(isa)
    , simd_w_(vlen / sizeof(float))
    , bcast_(bcast)
    , tail_size_(calc_tail_size(tensor_md))
    , tensor_md_(tensor_md)
    , number_vmm_single_compute_(number_vmm_single_compute) {}

size_t jit_prelu_base_kernel_t::simd_w() const noexcept {
    return simd_w_;
}

prelu::bcast jit_prelu_base_kernel_t::get_bcast() const noexcept {
    return bcast_;
}

void jit_prelu_base_kernel_t::generate() {
    Xbyak::Label unroll_loop, unroll_loop_tail, nelems_tail, end;
    const auto unrolling_factor = calc_unrolling_factor();

    preamble();
    load_kernel_call_params();
    prepare_kernel_const_vars();

    xor_(reg_offset_, reg_offset_);
    L(unroll_loop);
    {
        const size_t offt = unrolling_factor * simd_w_;
        cmp(reg_data_size_, offt);
        jl(unroll_loop_tail, T_NEAR);

        compute_dst(unrolling_factor, false /*tail*/);
        sub(reg_data_size_, offt);
        add(reg_offset_, offt);
        jmp(unroll_loop);
    }

    static constexpr size_t single_unrolling = 1u;
    L(unroll_loop_tail);
    {
        cmp(reg_data_size_, simd_w_);
        jl(nelems_tail, T_NEAR);

        compute_dst(single_unrolling, false /*tail*/);
        sub(reg_data_size_, simd_w_);
        add(reg_offset_, simd_w_);
        jmp(unroll_loop_tail);
    }

    L(nelems_tail);
    {
        cmp(reg_data_size_, 1);
        jl(end, T_NEAR);

        compute_dst(single_unrolling, true /*tail*/);
    }

    L(end);
    finalize();

    postamble();
}

size_t jit_prelu_base_kernel_t::calc_tail_size(
        const memory_desc_wrapper &tensor_md) const noexcept {

    const auto &ndims = tensor_md.ndims();
    dim_t nelems = 0;
    if (bcast_ == prelu::bcast::full)
        nelems = tensor_md.nelems();
    else if (bcast_ == prelu::bcast::per_oc_n_spatial_c)
        nelems = tensor_md.dims()[1];
    else if (bcast_ == prelu::bcast::per_oc_n_c_spatial && ndims >= 3)
        nelems = utils::array_product(tensor_md.dims() + 2, ndims - 2);

    return nelems % simd_w_;
}

int jit_prelu_base_kernel_t::reserve_vmm() {
    return number_reserved_vmms_++;
}

size_t jit_prelu_base_kernel_t::get_number_reserved_vmms() const noexcept {
    static constexpr size_t number_vmm_reserved_bf16_process = 4u;

    const bool process_bf16_with_emu = any_tensor_bf16() && isa_ == avx512_core;

    return number_reserved_vmms_
            + (process_bf16_with_emu ? number_vmm_reserved_bf16_process : 0);
}

int jit_prelu_base_kernel_t::get_compute_vmm(
        size_t base_idx, size_t unroll_group) const {
    return number_reserved_vmms_ + base_idx
            + unroll_group * number_vmm_single_compute_;
}

size_t jit_prelu_base_kernel_t::calc_unrolling_factor() const noexcept {
    const auto n_vregs = prelu::get_n_vregs(isa_);
    const size_t number_of_available_regs
            = n_vregs - get_number_reserved_vmms();
    const size_t max_unrolling_factor
            = number_of_available_regs / number_vmm_single_compute_;

    size_t single_thread_estimated_elems = 0;
    const auto &dims = tensor_md_.dims();
    const auto &ndims = tensor_md_.ndims();
    const dim_t D = ndims >= 5 ? dims[ndims - 3] : 1;
    const dim_t H = ndims >= 4 ? dims[ndims - 2] : 1;
    const dim_t W = ndims >= 3 ? dims[ndims - 1] : 1;
    const dim_t SP = D * H * W;

    if (bcast_ == prelu::bcast::full) {
        const size_t nelems = tensor_md_.nelems();
        single_thread_estimated_elems = nelems / dnnl_get_max_threads();
    } else if (bcast_ == prelu::bcast::per_oc_n_spatial_c) {
        single_thread_estimated_elems = tensor_md_.dims()[1];
    } else if (bcast_ == prelu::bcast::per_oc_blocked) {
        single_thread_estimated_elems = SP * simd_w_;
    } else if (bcast_ == prelu::bcast::per_oc_n_c_spatial) {
        single_thread_estimated_elems = SP;
    }

    const size_t estimated_vectors_used = nstl::max(
            static_cast<size_t>(
                    std::floor(single_thread_estimated_elems / simd_w_)),
            static_cast<size_t>(1));

    return nstl::min(max_unrolling_factor, estimated_vectors_used);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
