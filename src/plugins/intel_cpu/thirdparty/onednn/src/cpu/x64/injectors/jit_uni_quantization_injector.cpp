/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::init_crop_ptrs(const Xbyak::RegExp& ptr_begin, const Xbyak::Operand& ch_off) {
    h->mov(reg_d_weights_, h->ptr[ptr_begin]);
    h->mov(reg_d_bias_, h->ptr[ptr_begin]);

    if (post_op_.quantization.per_channel[post_op_.quantization.crop_low] && !post_op_.quantization.all_default[post_op_.quantization.crop_low])
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.per_channel[post_op_.quantization.crop_high] && !post_op_.quantization.all_default[post_op_.quantization.crop_high])
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_crop_impl(const std::set<size_t>& vmmIdxs, int offset, bool is_scalar, bool is_broadcast) {
    size_t weights_off =  post_op_.quantization.offset[post_op_.quantization.crop_low] * sizeof(float);
    size_t bias_off =  post_op_.quantization.offset[post_op_.quantization.crop_high] * sizeof(float);

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.crop_low])
            h->uni_vmovss(xmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.crop_low])
            h->uni_vpxor(vmm_d_weights_, vmm_d_weights_, vmm_d_weights_);
        else
            h->uni_vmovss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.crop_low])
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.crop_low])
            h->uni_vpxor(vmm_d_weights_, vmm_d_weights_, vmm_d_weights_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (auto vmmIdx : vmmIdxs) {
            Vmm vmm_dst = Vmm(vmmIdx);
            h->uni_vmaxps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.crop_high])
            h->uni_vmovss(xmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.crop_high])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->uni_vmovss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.crop_high])
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.crop_high])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    }

    for (auto vmmIdx : vmmIdxs) {
        Vmm vmm_dst = Vmm(vmmIdx);

        if (vmm_d_weights_.getIdx() != vmm_d_bias_.getIdx())
            h->uni_vmaxps(vmm_dst, vmm_dst, vmm_d_weights_);

        h->uni_vminps(vmm_dst, vmm_dst, vmm_d_bias_);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_crop(const std::set<size_t>& vmmIdxs, int offset, bool is_scalar, bool is_broadcast) {
    compute_crop_impl(vmmIdxs, offset, is_scalar, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_crop(int start_idx, int end_idx, int offset, bool is_scalar, bool is_broadcast) {
    std::set<size_t> vmmIdxs;
    for (int i = start_idx; i < end_idx; i++) {
        vmmIdxs.insert(i);
    }

    compute_crop_impl(vmmIdxs, offset, is_scalar, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::init_input_scale_shift_ptrs(const Xbyak::RegExp& ptr_begin, const Xbyak::Operand& ch_off) {
    h->mov(reg_d_weights_, h->ptr[ptr_begin]);
    h->mov(reg_d_bias_, h->ptr[ptr_begin]);

    if (post_op_.quantization.per_channel[post_op_.quantization.inp_scale])
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.per_channel[post_op_.quantization.inp_shift] && !post_op_.quantization.all_default[post_op_.quantization.inp_shift])
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_input_scale_shift_impl(
        const std::set<size_t>& vmmIdxs, int offset, bool do_rounding, bool is_scalar, bool is_broadcast) {
    size_t weights_off =  post_op_.quantization.offset[post_op_.quantization.inp_scale] * sizeof(float);
    size_t bias_off =  post_op_.quantization.offset[post_op_.quantization.inp_shift] * sizeof(float);

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.inp_scale])
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.inp_scale])
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (auto vmmIdx : vmmIdxs) {
            Vmm vmm_dst = Vmm(vmmIdx);

            h->uni_vmulps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.inp_shift])
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.inp_shift])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.inp_shift])
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.inp_shift])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    }

    for (auto vmmIdx : vmmIdxs) {
        Vmm vmm_dst = Vmm(vmmIdx);

        if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx())
            h->uni_vaddps(vmm_dst, vmm_dst, vmm_d_bias_);
        else
            h->uni_vfmadd213ps(vmm_dst, vmm_d_weights_, vmm_d_bias_);

        if (do_rounding)
            h->uni_vroundps(vmm_dst, vmm_dst, 0);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_input_scale_shift(int start_idx, int end_idx, int offset, bool do_rounding, bool is_scalar, bool is_broadcast) {
    std::set<size_t> vmmIdxs;
    for (int i = start_idx; i < end_idx; i++) {
        vmmIdxs.insert(i);
    }

    compute_input_scale_shift_impl(vmmIdxs, offset, do_rounding, is_scalar, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_input_scale_shift(const std::set<size_t>& vmmIdxs, int offset, bool do_rounding, bool is_scalar, bool is_broadcast) {
    compute_input_scale_shift_impl(vmmIdxs, offset, do_rounding, is_scalar, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::init_output_scale_shift_ptrs(const Xbyak::RegExp& ptr_begin, const Xbyak::Operand& ch_off) {
    if (!do_dequantization)
        return;

    h->mov(reg_d_weights_, h->ptr[ptr_begin]);
    h->mov(reg_d_bias_, h->ptr[ptr_begin]);

    if (post_op_.quantization.per_channel[post_op_.quantization.output_scale])
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.per_channel[post_op_.quantization.output_shift] && !post_op_.quantization.all_default[post_op_.quantization.output_shift])
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_output_scale_shift_impl(const std::set<size_t>& vmmIdxs, int offset, bool is_scalar, bool is_broadcast) {
    size_t weights_off =  post_op_.quantization.offset[post_op_.quantization.output_scale] * sizeof(float);
    size_t bias_off =  post_op_.quantization.offset[post_op_.quantization.output_shift] * sizeof(float);

    if (!do_dequantization)
        return;

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.output_scale])
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.output_scale])
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + weights_off]);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset + weights_off]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (auto &vmmIdx : vmmIdxs) {
            Vmm vmm_dst = Vmm(vmmIdx);

            h->uni_vmulps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (!post_op_.quantization.per_channel[post_op_.quantization.output_shift])
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.output_shift])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    } else {
        if (!post_op_.quantization.per_channel[post_op_.quantization.output_shift])
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + bias_off]);
        else if (post_op_.quantization.all_default[post_op_.quantization.output_shift])
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset + bias_off]);
    }

    for (auto &vmmIdx : vmmIdxs) {
        Vmm vmm_dst = Vmm(vmmIdx);

        if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx())
            h->uni_vaddps(vmm_dst, vmm_dst, vmm_d_bias_);
        else
            h->uni_vfmadd213ps(vmm_dst, vmm_d_weights_, vmm_d_bias_);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_output_scale_shift(int start_idx, int end_idx, int offset, bool is_scalar, bool is_broadcast) {
    std::set<size_t> vmmIdxs;
    for (int i = start_idx; i < end_idx; i++) {
        vmmIdxs.insert(i);
    }

    compute_output_scale_shift_impl(vmmIdxs, offset, is_scalar, is_broadcast);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_quantization_injector_f32<isa, Vmm>::compute_output_scale_shift(const std::set<size_t>& vmmIdxs, int offset, bool is_scalar, bool is_broadcast) {
    compute_output_scale_shift_impl(vmmIdxs, offset, is_scalar, is_broadcast);
}

template struct jit_uni_quantization_injector_f32<avx512_core_bf16>;
template struct jit_uni_quantization_injector_f32<avx512_core>;
template struct jit_uni_quantization_injector_f32<avx512_core, Xbyak::Ymm>;
template struct jit_uni_quantization_injector_f32<avx512_core, Xbyak::Xmm>;
template struct jit_uni_quantization_injector_f32<avx512_common>;
template struct jit_uni_quantization_injector_f32<avx512_common, Xbyak::Ymm>;
template struct jit_uni_quantization_injector_f32<avx2, Xbyak::Ymm>;
template struct jit_uni_quantization_injector_f32<avx2, Xbyak::Xmm>;
template struct jit_uni_quantization_injector_f32<avx, Xbyak::Ymm>;
template struct jit_uni_quantization_injector_f32<avx, Xbyak::Xmm>;
template struct jit_uni_quantization_injector_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
