/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_uni_quantization.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::init_crop_ptrs(const Xbyak::Operand& ch_off) {
    h->mov(reg_d_weights_, reinterpret_cast<size_t>(post_op_.quantization.crop_low_data->shifts_));
    h->mov(reg_d_bias_, reinterpret_cast<size_t>(post_op_.quantization.crop_high_data->shifts_));

    if (post_op_.quantization.crop_low_data->count_ != 1 && !post_op_.quantization.crop_low_data->has_default_values())
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.crop_high_data->count_ != 1  && !post_op_.quantization.crop_high_data->has_default_values())
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::compute_crop(int start_idx, int end_idx, int offset, bool is_scalar, bool is_broadcast) {
    if (is_scalar) {
        if (post_op_.quantization.crop_low_data->count_ == 1)
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_]);
        else if (post_op_.quantization.crop_low_data->has_default_values())
            h->uni_vpxor(vmm_d_weights_, vmm_d_weights_, vmm_d_weights_);
        else
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    } else {
        if (post_op_.quantization.crop_low_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_]);
        else if (post_op_.quantization.crop_low_data->has_default_values())
            h->uni_vpxor(vmm_d_weights_, vmm_d_weights_, vmm_d_weights_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (int jj = start_idx; jj < end_idx; jj++) {
            Vmm vmm_dst = Vmm(jj);
            h->uni_vmaxps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (post_op_.quantization.crop_high_data->count_ == 1)
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.crop_high_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    } else {
        if (post_op_.quantization.crop_high_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.crop_high_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    }

    for (int jj = start_idx; jj < end_idx; jj++) {
        Vmm vmm_dst = Vmm(jj);

        if (vmm_d_weights_.getIdx() != vmm_d_bias_.getIdx())
            h->uni_vmaxps(vmm_dst, vmm_dst, vmm_d_weights_);

        h->uni_vminps(vmm_dst, vmm_dst, vmm_d_bias_);
    }
}

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::init_input_scale_shift_ptrs(const Xbyak::Operand& ch_off) {
    h->mov(reg_d_weights_, reinterpret_cast<size_t>(post_op_.quantization.input_scale_data->scales_));
    h->mov(reg_d_bias_, reinterpret_cast<size_t>(post_op_.quantization.input_shift_data->shifts_));

    if (post_op_.quantization.input_scale_data->count_ != 1)
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.input_shift_data->count_ != 1 && !post_op_.quantization.input_shift_data->has_default_values())
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::compute_input_scale_shift(int start_idx, int end_idx, int offset, bool do_rounding, bool is_scalar, bool is_broadcast) {
    if (is_scalar) {
        if (post_op_.quantization.input_scale_data->count_ == 1)
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_]);
        else
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    } else {
        if (post_op_.quantization.input_scale_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_]);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (int jj = start_idx; jj < end_idx; jj++) {
            Vmm vmm_dst = Vmm(jj);

            h->uni_vmulps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (post_op_.quantization.input_shift_data->count_ == 1)
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.input_shift_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    } else {
        if (post_op_.quantization.input_shift_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.input_shift_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    }

    for (int jj = start_idx; jj < end_idx; jj++) {
        Vmm vmm_dst = Vmm(jj);

        if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx())
            h->uni_vaddps(vmm_dst, vmm_dst, vmm_d_bias_);
        else
            h->uni_vfmadd213ps(vmm_dst, vmm_d_weights_, vmm_d_bias_);

        if (do_rounding)
            h->uni_vroundps(vmm_dst, vmm_dst, 0);
    }
}

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::init_output_scale_shift_ptrs(const Xbyak::Operand& ch_off) {
    if (!do_dequantization)
        return;

    h->mov(reg_d_weights_, reinterpret_cast<size_t>(post_op_.quantization.output_scale_data->scales_));
    h->mov(reg_d_bias_, reinterpret_cast<size_t>(post_op_.quantization.output_shift_data->shifts_));

    if (post_op_.quantization.output_scale_data->count_ != 1)
        h->add(reg_d_weights_, ch_off);
    if (post_op_.quantization.output_shift_data->count_ != 1 && !post_op_.quantization.output_shift_data->has_default_values())
        h->add(reg_d_bias_, ch_off);
}

template <cpu_isa_t isa>
void jit_uni_quantization_injector_f32<isa>::compute_output_scale_shift(int start_idx, int end_idx, int offset, bool is_scalar, bool is_broadcast) {
    if (!do_dequantization)
        return;

    if (is_scalar) {
        if (post_op_.quantization.output_scale_data->count_ == 1)
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_]);
        else
            h->movss(xmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    } else {
        if (post_op_.quantization.output_scale_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_]);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
        else
            h->uni_vmovups(vmm_d_weights_, h->ptr[reg_d_weights_ + offset]);
    }

    if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx()) {
        for (int jj = start_idx; jj < end_idx; jj++) {
            Vmm vmm_dst = Vmm(jj);

            h->uni_vmulps(vmm_dst, vmm_dst, vmm_d_weights_);
        }
    }

    if (is_scalar) {
        if (post_op_.quantization.output_shift_data->count_ == 1)
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.output_shift_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else
            h->movss(xmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    } else {
        if (post_op_.quantization.output_shift_data->count_ == 1)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_]);
        else if (post_op_.quantization.output_shift_data->has_default_values())
            h->uni_vpxor(vmm_d_bias_, vmm_d_bias_, vmm_d_bias_);
        else if (is_broadcast)
            h->uni_vbroadcastss(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
        else
            h->uni_vmovups(vmm_d_bias_, h->ptr[reg_d_bias_ + offset]);
    }

    for (int jj = start_idx; jj < end_idx; jj++) {
        Vmm vmm_dst = Vmm(jj);

        if (vmm_d_weights_.getIdx() == vmm_d_bias_.getIdx())
            h->uni_vaddps(vmm_dst, vmm_dst, vmm_d_bias_);
        else
            h->uni_vfmadd213ps(vmm_dst, vmm_d_weights_, vmm_d_bias_);
    }
}

template struct jit_uni_quantization_injector_f32<avx512_core>;
template struct jit_uni_quantization_injector_f32<avx512_common>;
template struct jit_uni_quantization_injector_f32<avx2>;
template struct jit_uni_quantization_injector_f32<sse42>;

struct jit_args {
    const uint8_t* from;
    const uint8_t* to;
    const float* thresholds;
    const float* output_mask;

    const float* crop_low;
    const float* crop_high;
    const float* input_scale;
    const float* input_shift;
    const float* output_scale;
    const float* output_shift;

    size_t src_step;
    size_t dst_step;
    size_t block_size;
    size_t work_amount;
};

struct jit_uni_quantization_kernel : public c_compatible {
    const quantization_desc_t &desc_;
    void (*ker_)(const jit_args *);

    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_quantization_kernel(const quantization_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_quantization_kernel() {}
};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_bin_depthwise_kernel : public jit_uni_quantization_kernel,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bin_depthwise_kernel)
    jit_uni_bin_depthwise_kernel(const quantization_desc_t &desc)
        : jit_uni_quantization_kernel(desc), jit_generator() {
        assert(one_of(desc.alg_kind, alg_kind::binarization_depthwise));
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_thresholds, ptr[param + GET_OFF(thresholds)]);
        mov(reg_output_mask, ptr[param + GET_OFF(output_mask)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        const int nbits = 8;
        int simd_w = isa == avx512_common ? 16 : 8;
        const int C = desc.src_desc.dims[1];
        const int tail_size = C % simd_w;

        Label unrolled_loop_label;
        Label main_loop_label;
        Label tail_label;
        Label exit_label;

        L(unrolled_loop_label); {
            int step = isa == sse42 ? nbits / 2 : isa == avx2 ? nbits : 2 * nbits;
            const int ur_ch = isa == sse42 ? nbits : isa == avx2 ? nbits / 2 : nbits / 4;
            const int unrolled_loop_step = ur_ch * step;

            cmp(reg_work_amount, unrolled_loop_step);
            jl(main_loop_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int ch = 0; ch < ur_ch; ch++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + ch*step*sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + ch*step*sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + ch*step*sizeof(float)]);
                if (isa == avx512_common) {
                    vcmpps(k_mask0, vmm_src(0), vmm_wei(0), _cmp_gt_os);
                    vptestmd(k_mask1, vmm_mask(0), vmm_mask(0));
                    kxnorw(k_mask0, k_mask0, k_mask1);
                    kmovw(reg_src_32, k_mask0);
                } else {
                    uni_vcmpgtps(vmm_src(0), vmm_src(0), vmm_wei(0));
                    uni_vpcmpeqd(vmm_src(0), vmm_src(0), vmm_mask(0));
                    uni_vmovmskps(reg_src_32, vmm_src(0));
                }
                shl(reg_src_32, ch * step);
                or_(reg_bin_32, reg_src_32);
            }
            mov(ptr[reg_to], reg_bin_32);

            add(reg_from, unrolled_loop_step*sizeof(float));
            add(reg_thresholds, unrolled_loop_step*sizeof(float));
            add(reg_output_mask, unrolled_loop_step*sizeof(float));
            add(reg_to, sizeof(uint32_t));
            sub(reg_work_amount, unrolled_loop_step);

            jmp(unrolled_loop_label, T_NEAR);
        }

        L(main_loop_label); {
            int repeats = isa == sse42 ? 2 : 1;
            int step = isa == sse42 ? nbits / 2 : isa == avx2 ? nbits : nbits * 2;
            const int main_loop_step = step * repeats;

            cmp(reg_work_amount, main_loop_step);
            jl(tail_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + i*step*sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + i*step*sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + i*step*sizeof(float)]);
                if (isa == avx512_common) {
                    vcmpps(k_mask0, vmm_src(0), vmm_wei(0), _cmp_gt_os);
                    vptestmd(k_mask1, vmm_mask(0), vmm_mask(0));
                    kxnorw(k_mask0, k_mask0, k_mask1);
                    kmovw(reg_src_32, k_mask0);
                } else {
                    uni_vcmpgtps(vmm_src(0), vmm_src(0), vmm_wei(0));
                    uni_vpcmpeqd(vmm_src(0), vmm_src(0), vmm_mask(0));
	                uni_vmovmskps(reg_src_32, vmm_src(0));
                }
                shl(reg_src_32, i * step);
                or_(reg_bin_32, reg_src_32);
            }
            if (isa == avx512_common)
                mov(ptr[reg_to], reg_bin_16);
            else
                mov(ptr[reg_to], reg_bin_8);

            add(reg_from, main_loop_step*sizeof(float));
            add(reg_thresholds, main_loop_step*sizeof(float));
            add(reg_output_mask, main_loop_step*sizeof(float));
            add(reg_to, isa == avx512_common ? sizeof(uint16_t) : sizeof(uint8_t));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_label); {
            if (tail_size != 0) {
                xor_(reg_bin_32, reg_bin_32);
                mov(reg_mask, 1);
                for (int c = 0; c < tail_size; c++) {
                    uni_vpxor(xmm_src(0), xmm_src(0), xmm_src(0));
                    uni_vpxor(xmm_wei(0), xmm_wei(0), xmm_wei(0));
                    uni_vpxor(xmm_mask(0), xmm_mask(0), xmm_mask(0));

                    movss(xmm_src(0), ptr[reg_from + c * sizeof(float)]);
                    movss(xmm_wei(0), ptr[reg_thresholds + c * sizeof(float)]);
                    movss(xmm_mask(0), ptr[reg_output_mask + c * sizeof(float)]);
                    uni_vcmpgtps(xmm_src(0), xmm_src(0), xmm_wei(0));
                    uni_vpcmpeqd(xmm_src(0), xmm_src(0), xmm_mask(0));
                    uni_vmovmskps(reg_src_32, xmm_src(0));

                    shl(reg_src_32, c);
                    and_(reg_src_32, reg_mask);
                    or_(reg_bin_32, reg_src_32);
                    shl(reg_mask, 1);
                }
                if (isa == avx512_common && tail_size > nbits)
                	mov(ptr[reg_to], reg_bin_16);
                else
                	mov(ptr[reg_to], reg_bin_8);
            }
        }

        L(exit_label);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm vmm_src(int idx) { return Vmm(idx); }
    inline Xmm xmm_src(int idx) { return Xmm(idx); }
    inline Vmm vmm_wei(int idx) { return Vmm(idx + 4); }
    inline Vmm vmm_mask(int idx) { return Vmm(idx + 5); }
    inline Xmm xmm_wei(int idx) { return Xmm(idx + 4); }
    inline Xmm xmm_mask(int idx) { return Xmm(idx + 5); }

    Reg64 param = abi_param1;
    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_thresholds = r11;
    Reg64 reg_output_mask = r14;
    Reg16 reg_bin_16 = r12w;
    Reg32 reg_bin_32 = r12d;
    Reg8 reg_bin_8 = r12b;
    Reg32 reg_src_32 = r13d;
    Reg32 reg_mask = r15d;

    const unsigned char _cmp_gt_os = 6;
    Xbyak::Opmask k_mask0 = Xbyak::Opmask(1);
    Xbyak::Opmask k_mask1 = Xbyak::Opmask(2);
};

template <cpu_isa_t isa>
struct jit_uni_quant_depthwise_kernel : public jit_uni_quantization_kernel,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bin_depthwise_kernel)
    jit_uni_quant_depthwise_kernel(const quantization_desc_t &desc)
        : jit_uni_quantization_kernel(desc), jit_generator() {
        assert(one_of(desc.alg_kind, alg_kind::quantization_quantize_dequantize, alg_kind::quantization_quantize));
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        src_data_type = desc.src_desc.data_type;
        wei_data_type = mkldnn_f32;
        dst_data_type = desc.dst_desc.data_type;

        do_dequantization = desc.alg_kind == alg_kind::quantization_quantize_dequantize;
        do_rounding = do_dequantization || dst_data_type == mkldnn_f32;

        this->preamble();

        if (desc.src_desc.format == tnc || desc.src_desc.format == nchw || desc.src_desc.format == ncdhw)
            compute_planar();
        else
            compute_generic(desc);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm vmm_val(int idx) { return Vmm(idx + 0); }
    inline Vmm vmm_crop_low(int idx) { return Vmm(idx + 2); }
    inline Vmm vmm_crop_high(int idx) { return Vmm(idx + 4); }
    inline Vmm vmm_input_scale(int idx) { return Vmm(idx + 6); }
    inline Vmm vmm_input_shift(int idx) { return Vmm(idx + 8); }
    inline Vmm vmm_output_scale(int idx) { return Vmm(idx + 10); }
    inline Vmm vmm_output_shift(int idx) { return Vmm(idx + 12); }

    inline Ymm ymm_val(int idx) { return Ymm(idx + 0); }
    inline Ymm ymm_crop_low(int idx) { return Ymm(idx + 2); }
    inline Ymm ymm_crop_high(int idx) { return Ymm(idx + 4); }
    inline Ymm ymm_input_scale(int idx) { return Ymm(idx + 6); }
    inline Ymm ymm_input_shift(int idx) { return Ymm(idx + 8); }
    inline Ymm ymm_output_scale(int idx) { return Ymm(idx + 10); }
    inline Ymm ymm_output_shift(int idx) { return Ymm(idx + 12); }

    inline Xmm xmm_val(int idx) { return Xmm(idx + 0); }
    inline Xmm xmm_crop_low(int idx) { return Xmm(idx + 2); }
    inline Xmm xmm_crop_high(int idx) { return Xmm(idx + 4); }
    inline Xmm xmm_input_scale(int idx) { return Xmm(idx + 6); }
    inline Xmm xmm_input_shift(int idx) { return Xmm(idx + 8); }
    inline Xmm xmm_output_scale(int idx) { return Xmm(idx + 10); }
    inline Xmm xmm_output_shift(int idx) { return Xmm(idx + 12); }

    Vmm vmm_zero = Vmm(14);

    Reg64 param = abi_param1;
    Reg64 reg_from = rbp;
    Reg64 reg_to = r9;
    Reg64 aux_reg_from = abi_not_param1;
    Reg64 aux_reg_to = r8;
    Reg64 reg_src_step = r10;
    Reg64 reg_dst_step = rsi;
    Reg64 reg_block_size = r11;
    Reg64 reg_work_amount = r12;

    Reg8 reg_tmp_8 = r9b;
    Reg32 reg_tmp_32 = r9d;
    Reg64 reg_tmp_64 = r9;

    Reg64 reg_crop_low = r13;
    Reg64 reg_crop_high = r14;
    Reg64 reg_input_scale = r15;
    Reg64 reg_input_shift = rax;
    Reg64 reg_output_scale = rbx;
    Reg64 reg_output_shift = rdx;

    mkldnn_data_type_t src_data_type;
    mkldnn_data_type_t dst_data_type;
    mkldnn_data_type_t wei_data_type;

    bool do_rounding;
    bool do_dequantization;

    inline void compute_planar() {
        int src_type_size = types::data_type_size(src_data_type);
        int dst_type_size = types::data_type_size(dst_data_type);

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);

        mov(reg_crop_low, ptr[param + GET_OFF(crop_low)]);
        mov(reg_crop_high, ptr[param + GET_OFF(crop_high)]);
        mov(reg_input_scale, ptr[param + GET_OFF(input_scale)]);
        mov(reg_input_shift, ptr[param + GET_OFF(input_shift)]);
        mov(reg_output_scale, ptr[param + GET_OFF(output_scale)]);
        mov(reg_output_shift, ptr[param + GET_OFF(output_shift)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (isa == avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int simd_w = isa == avx512_common ? 16 : 8;
        int tail_simd_w = 4;
        int repeats = isa == sse42 ? 2 : 1;

        Label main_loop_label;
        Label tail_blk4_label;
        Label tail_blk4_loop_label;
        Label tail_blk4_exit_label;
        Label tail_label;
        Label tail_loop_label;
        Label exit_label;

        uni_vbroadcastss(vmm_crop_low(0), ptr[reg_crop_low]);
        uni_vbroadcastss(vmm_crop_high(0), ptr[reg_crop_high]);
        uni_vbroadcastss(vmm_input_scale(0), ptr[reg_input_scale]);
        uni_vbroadcastss(vmm_input_shift(0), ptr[reg_input_shift]);
        if (do_dequantization) {
            uni_vbroadcastss(vmm_output_scale(0), ptr[reg_output_scale]);
            uni_vbroadcastss(vmm_output_shift(0), ptr[reg_output_shift]);
        }

        L(main_loop_label); {
            cmp(reg_work_amount, simd_w);
            jl(tail_blk4_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], src_data_type);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(0));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(0));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(0), vmm_input_shift(0));
                if (do_rounding) uni_vroundps(vmm_val(i), vmm_val(i), 0);
                if (do_dequantization) uni_vfmadd213ps(vmm_val(i), vmm_output_scale(0), vmm_output_shift(0));

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), dst_data_type);
            }

            sub(reg_work_amount, simd_w);
            add(reg_from, simd_w * src_type_size);
            add(reg_to, simd_w * dst_type_size);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_blk4_label); {
            cmp(reg_work_amount, tail_simd_w);
            jl(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[reg_from], src_data_type);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_vector(ptr[reg_to], xmm_val(0), dst_data_type);

            sub(reg_work_amount, tail_simd_w);
            add(reg_from, tail_simd_w * src_type_size);
            add(reg_to, tail_simd_w * dst_type_size);
        }

        L(tail_blk4_exit_label);

        mov(aux_reg_from, reg_from);
        mov(aux_reg_to, reg_to);

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            load_scalar(xmm_val(0), ptr[aux_reg_from], src_data_type);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_scalar(ptr[aux_reg_to], xmm_val(0), dst_data_type);

            sub(reg_work_amount, 1);
            add(aux_reg_from, 1 * src_type_size);
            add(aux_reg_to, 1 * dst_type_size);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    inline void compute_generic(const quantization_desc_t &desc) {
        int src_type_size = types::data_type_size(src_data_type);
        int wei_type_size = types::data_type_size(wei_data_type);
        int dst_type_size = types::data_type_size(dst_data_type);

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);

        mov(reg_crop_low, ptr[param + GET_OFF(crop_low)]);
        mov(reg_crop_high, ptr[param + GET_OFF(crop_high)]);
        mov(reg_input_scale, ptr[param + GET_OFF(input_scale)]);
        mov(reg_input_shift, ptr[param + GET_OFF(input_shift)]);
        if (do_dequantization) {
            mov(reg_output_scale, ptr[param + GET_OFF(output_scale)]);
            mov(reg_output_shift, ptr[param + GET_OFF(output_shift)]);
        }

        mov(reg_src_step, ptr[param + GET_OFF(src_step)]);
        mov(reg_dst_step, ptr[param + GET_OFF(dst_step)]);
        mov(reg_block_size, ptr[param + GET_OFF(block_size)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (isa == avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int simd_w = isa == avx512_common ? 16 : 8;
        int tail8_simd_w = 8;
        int tail4_simd_w = 4;
        int repeats = isa == sse42 ? 2 : 1;

        Label main_loop_label;
        Label tail_blk8_label;
        Label tail_blk8_loop_label;
        Label tail_blk8_exit_label;
        Label tail_blk4_label;
        Label tail_blk4_loop_label;
        Label tail_blk4_exit_label;
        Label tail_label;
        Label tail_loop_label;
        Label exit_label;

        cmp(reg_block_size, simd_w);
        jl(simd_w == 16 ? tail_blk8_label : tail_blk4_label, T_NEAR);

        for (int i = 0; i < repeats; i++) {
            uni_vmovups(vmm_crop_low(i), ptr[reg_crop_low + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_crop_high(i), ptr[reg_crop_high + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_input_scale(i), ptr[reg_input_scale + i * (simd_w / 2) * sizeof(float)]);
            uni_vmovups(vmm_input_shift(i), ptr[reg_input_shift + i * (simd_w / 2) * sizeof(float)]);
            if (do_dequantization) {
                uni_vmovups(vmm_output_scale(i), ptr[reg_output_scale + i * (simd_w / 2) * sizeof(float)]);
                uni_vmovups(vmm_output_shift(i), ptr[reg_output_shift + i * (simd_w / 2) * sizeof(float)]);
            }
        }

        L(main_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], src_data_type);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(i));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(i));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(i), vmm_input_shift(i));
                if (do_rounding) uni_vroundps(vmm_val(i), vmm_val(i), 0);
                if (do_dequantization) uni_vfmadd213ps(vmm_val(i), vmm_output_scale(i), vmm_output_shift(i));

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), dst_data_type);
            }

            dec(reg_work_amount);
            add(reg_from, reg_src_step);
            add(reg_to, reg_dst_step);

            jmp(main_loop_label, T_NEAR);
        }

        if (simd_w == 16) {
            L(tail_blk8_label);

            cmp(reg_block_size, tail8_simd_w);
            jl(tail_blk4_label, T_NEAR);

            mov(aux_reg_to, reg_to);
            mov(aux_reg_from, reg_from);
            mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

            uni_vmovups(ymm_crop_low(0), ptr[reg_crop_low]);
            uni_vmovups(ymm_crop_high(0), ptr[reg_crop_high]);
            uni_vmovups(ymm_input_scale(0), ptr[reg_input_scale]);
            uni_vmovups(ymm_input_shift(0), ptr[reg_input_shift]);
            if (do_dequantization) {
                uni_vmovups(ymm_output_scale(0), ptr[reg_output_scale]);
                uni_vmovups(ymm_output_shift(0), ptr[reg_output_shift]);
            }

            L(tail_blk8_loop_label); {
                cmp(reg_work_amount, 0);
                jle(tail_blk8_exit_label, T_NEAR);

                load_vector(ymm_val(0), ptr[aux_reg_from], src_data_type);

                uni_vminps(ymm_val(0), ymm_val(0), ymm_crop_high(0));
                uni_vmaxps(ymm_val(0), ymm_val(0), ymm_crop_low(0));
                uni_vfmadd213ps(ymm_val(0), ymm_input_scale(0), ymm_input_shift(0));
                if (do_rounding) uni_vroundps(ymm_val(0), ymm_val(0), 0);
                if (do_dequantization) uni_vfmadd213ps(ymm_val(0), ymm_output_scale(0), ymm_output_shift(0));

                store_vector(ptr[aux_reg_to], ymm_val(0), dst_data_type);

                dec(reg_work_amount);
                add(aux_reg_from, reg_src_step);
                add(aux_reg_to, reg_dst_step);

                jmp(tail_blk8_loop_label, T_NEAR);
            }

            L(tail_blk8_exit_label);

            add(reg_from, tail8_simd_w * src_type_size);
            add(reg_to, tail8_simd_w * dst_type_size);
            add(reg_crop_low, tail8_simd_w * wei_type_size);
            add(reg_crop_high, tail8_simd_w * wei_type_size);
            add(reg_input_scale, tail8_simd_w * wei_type_size);
            add(reg_input_shift, tail8_simd_w * wei_type_size);
            if (do_dequantization) {
                add(reg_output_scale, tail8_simd_w * wei_type_size);
                add(reg_output_shift, tail8_simd_w * wei_type_size);
            }
            sub(reg_block_size, tail8_simd_w);
        }

        L(tail_blk4_label);

        cmp(reg_block_size, tail4_simd_w);
        jl(tail_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        uni_vmovups(xmm_crop_low(0), ptr[reg_crop_low]);
        uni_vmovups(xmm_crop_high(0), ptr[reg_crop_high]);
        uni_vmovups(xmm_input_scale(0), ptr[reg_input_scale]);
        uni_vmovups(xmm_input_shift(0), ptr[reg_input_shift]);
        if (do_dequantization) {
            uni_vmovups(xmm_output_scale(0), ptr[reg_output_scale]);
            uni_vmovups(xmm_output_shift(0), ptr[reg_output_shift]);
        }

        L(tail_blk4_loop_label); {
            cmp(reg_work_amount, 0);
            jle(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[aux_reg_from], src_data_type);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
            if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

            store_vector(ptr[aux_reg_to], xmm_val(0), dst_data_type);

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_blk4_loop_label, T_NEAR);
        }

        L(tail_blk4_exit_label);

        add(reg_from, tail4_simd_w * src_type_size);
        add(reg_to, tail4_simd_w * dst_type_size);
        add(reg_crop_low, tail4_simd_w * wei_type_size);
        add(reg_crop_high, tail4_simd_w * wei_type_size);
        add(reg_input_scale, tail4_simd_w * wei_type_size);
        add(reg_input_shift, tail4_simd_w * wei_type_size);
        if (do_dequantization) {
            add(reg_output_scale, tail4_simd_w * wei_type_size);
            add(reg_output_shift, tail4_simd_w * wei_type_size);
        }

        L(tail_label);

        cmp(reg_block_size, 0);
        jle(exit_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            for (int i = 0; i < desc.src_desc.dims[1] % tail4_simd_w; i++) {
                movss(xmm_crop_low(0), ptr[reg_crop_low + i * wei_type_size]);
                movss(xmm_crop_high(0), ptr[reg_crop_high + i * wei_type_size]);
                movss(xmm_input_scale(0), ptr[reg_input_scale + i * wei_type_size]);
                movss(xmm_input_shift(0), ptr[reg_input_shift + i * wei_type_size]);
                if (do_dequantization) {
                    movss(xmm_output_scale(0), ptr[reg_output_scale + i * wei_type_size]);
                    movss(xmm_output_shift(0), ptr[reg_output_shift + i * wei_type_size]);
                }

                load_scalar(xmm_val(0), ptr[aux_reg_from + i * src_type_size], src_data_type);

                uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
                uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
                uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
                if (do_rounding) uni_vroundps(xmm_val(0), xmm_val(0), 0);
                if (do_dequantization) uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));

                store_scalar(ptr[aux_reg_to + i * dst_type_size], xmm_val(0), dst_data_type);
            }

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    inline void load_vector(Zmm zmm_src, const Xbyak::Address &op, data_type_t src_dt) {
        switch (src_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(zmm_src, op);
                break;
            case data_type::s8:
                uni_vpmovsxbd(zmm_src, op);
                break;
            case data_type::u8:
                uni_vpmovzxbd(zmm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(zmm_src, zmm_src);
        }
    }

    inline void load_vector(Ymm ymm_src, const Xbyak::Address &op, data_type_t src_dt) {
        switch (src_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(ymm_src, op);
                break;
            case data_type::s8:
                uni_vpmovsxbd(ymm_src, op);
                break;
            case data_type::u8:
                uni_vpmovzxbd(ymm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(ymm_src, ymm_src);
        }
    }

    inline void load_vector(Xmm xmm_src, const Xbyak::Address &op, data_type_t src_dt) {
        switch (src_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(xmm_src, op);
                break;
            case data_type::s8:
                uni_vpmovsxbd(xmm_src, op);
                break;
            case data_type::u8:
                uni_vpmovzxbd(xmm_src, op);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, data_type_t src_dt) {
        switch (src_dt) {
            case data_type::f32:
            case data_type::s32:
                movss(xmm_src, op);
                break;
            case data_type::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case data_type::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown dst_dt");
        }

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Zmm zmm_dst, data_type_t dst_dt) {
        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(zmm_dst, zmm_dst);
        }

        switch (dst_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(op, zmm_dst);
                break;
            case data_type::s8:
                vpmovsdb(op, zmm_dst);
                break;
            case data_type::u8:
                vpmaxsd(zmm_dst, zmm_dst, vmm_zero);
                vpmovusdb(op, zmm_dst);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Ymm ymm_dst, data_type_t dst_dt) {
        Xmm xmm_dst = Xmm(ymm_dst.getIdx());

        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(ymm_dst, ymm_dst);
        }

        switch (dst_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(op, ymm_dst);
                break;
            case data_type::s8:
                uni_vpackssdw(ymm_dst, ymm_dst, ymm_dst);

                vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpacksswb(ymm_dst, ymm_dst, ymm_dst);

                vmovq(op, xmm_dst);
                break;
            case data_type::u8:
                uni_vpackusdw(ymm_dst, ymm_dst, ymm_dst);

                vpermq(ymm_dst, ymm_dst, 0x08);

                uni_vpackuswb(ymm_dst, ymm_dst, ymm_dst);

                vmovq(op, xmm_dst);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Xmm xmm_dst, data_type_t dst_dt) {
        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case data_type::f32:
            case data_type::s32:
                uni_vmovups(op, xmm_dst);
                break;
            case data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            case data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movd(op, xmm_dst);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, data_type_t dst_dt) {
        if (dst_dt != data_type::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case data_type::f32:
            case data_type::s32:
                movss(op, xmm_dst);
                break;
            case data_type::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case data_type::u8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }
};

} /* namespace */

template <cpu_isa_t isa>
void jit_uni_quantization_fwd_t<isa>::execute_binarization_forward() const {
    auto src = reinterpret_cast<const uint8_t*>(this->input_memory(0));
    auto thresholds = reinterpret_cast<const float*>(this->input_memory(1));
    auto output_mask = reinterpret_cast<const float*>(this->input_memory(2));
    auto dst = reinterpret_cast<uint8_t*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper thresholds_d(pd()->weights_pd(0));
    const memory_desc_wrapper output_mask_d(pd()->weights_pd(1));

    const int N = src_d.dims()[0];
    const int C = src_d.dims()[1];
    const int H = src_d.dims()[2];
    const int W = src_d.dims()[3];

    int nbits = 8;

    parallel_nd(N, H, W, [&](int n, int h, int w) {
	    auto arg = jit_args();

        arg.from    = &src[src_d.blk_off(n, 0, h, w) * sizeof(float)];
        arg.to      = &dst[dst_d.blk_off(n, 0, h, w) / nbits];
        arg.thresholds = &thresholds[thresholds_d.blk_off(0)];
        arg.output_mask = &output_mask[output_mask_d.blk_off(0)];
        arg.work_amount = (size_t)C;

        (*kernel_)(&arg);
    });
}

template <cpu_isa_t isa>
void jit_uni_quantization_fwd_t<isa>::execute_quantization_forward() const {
    auto src = reinterpret_cast<const uint8_t*>(this->input_memory(0));
    auto crop_low = reinterpret_cast<const float*>(this->input_memory(1));
    auto crop_high = reinterpret_cast<const float*>(this->input_memory(2));
    auto input_scale = reinterpret_cast<const float*>(this->input_memory(3));
    auto input_shift = reinterpret_cast<const float*>(this->input_memory(4));
    auto output_scale = reinterpret_cast<const float*>(this->input_memory(5));
    auto output_shift = reinterpret_cast<const float*>(this->input_memory(6));
    auto dst = reinterpret_cast<uint8_t*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());

    bool is_blk_format = src_d.format() != nhwc && src_d.format() != ndhwc;
    int blk_size = (src_d.format() == tnc || src_d.format() == nchw || src_d.format() == ncdhw) ? 1 : isa == avx512_common ? 16 : 8;

    auto src_data_type = src_d.data_type();
    auto src_type_size = types::data_type_size(src_data_type);

    auto dst_data_type = dst_d.data_type();
    auto dst_type_size = types::data_type_size(dst_data_type);

    const int N = src_d.dims()[0];
    const int C = src_d.dims()[1];
    const int CB = div_up(C, blk_size);
    const int D = src_d.ndims() == 5 ? src_d.dims()[2] : 1;
    const int H = src_d.ndims() == 3 ? src_d.dims()[2] : src_d.ndims() > 3 ? src_d.dims()[src_d.ndims() - 2] : 1;
    const int W = src_d.ndims() > 3 ? src_d.dims()[src_d.ndims() - 1] : 1;

    if (src_d.format() == tnc) {
        parallel_nd(N, CB, D, [&](int n, int cb, int d) {
            auto arg = jit_args();

            int c = cb * blk_size;

            size_t data_off = src_d.off(n, c, 0);


            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = &crop_low[c];
            arg.crop_high = &crop_high[c];
            arg.input_scale = &input_scale[c];
            arg.input_shift = &input_shift[c];
            arg.output_scale = &output_scale[c];
            arg.output_shift = &output_shift[c];

            arg.src_step = (size_t) blk_size * src_type_size;
            arg.dst_step = (size_t) blk_size * dst_type_size;
            arg.block_size = (size_t) blk_size;
            arg.work_amount = (size_t)H;

            (*kernel_)(&arg);
        });
    } else {
        parallel_nd(N, CB, D, H, [&](int n, int cb, int d, int h) {
            auto arg = jit_args();

            int c = cb * blk_size;

            size_t data_off = src_d.ndims() == 2 ? src_d.off(n, c) :
                              src_d.ndims() == 3 ? src_d.off(n, c, h) :
                              src_d.ndims() == 4 ? src_d.off(n, c, h, 0) :
                              src_d.off(n, c, d, h, 0);

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = &crop_low[c];
            arg.crop_high = &crop_high[c];
            arg.input_scale = &input_scale[c];
            arg.input_shift = &input_shift[c];
            arg.output_scale = &output_scale[c];
            arg.output_shift = &output_shift[c];

            arg.src_step = is_blk_format ? (size_t) blk_size * src_type_size : (size_t) C * src_type_size;
            arg.dst_step = is_blk_format ? (size_t) blk_size * dst_type_size : (size_t) C * dst_type_size;
            arg.block_size = (is_blk_format && src_d.format() != nc) ? (size_t) blk_size : nstl::min(blk_size, C - c);
            arg.work_amount = (size_t) W;

            (*kernel_)(&arg);
        });
    }
}

template <cpu_isa_t isa>
status_t jit_uni_quantization_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    auto desired_blk_fmt = isa == avx512_common ? nChw16c : nChw8c;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training, prop_kind::forward_inference)
        && IMPLICATION(utils::one_of(desc()->alg_kind, mkldnn_quantization_quantize_dequantize, mkldnn_quantization_quantize),
                       utils::one_of(desc()->src_desc.data_type, data_type::f32, data_type::u8, data_type::s8) &&
                       utils::everyone_is(data_type::f32, desc()->crop_low_desc.data_type, desc()->crop_high_desc.data_type,
                                          desc()->input_scale_desc.data_type, desc()->input_shift_desc.data_type,
                                          desc()->output_scale_desc.data_type, desc()->output_shift_desc.data_type) &&
                       utils::one_of(desc()->dst_desc.data_type, data_type::f32, data_type::u8, data_type::s8) &&
                       utils::everyone_is(x, desc()->crop_low_desc.format, desc()->crop_high_desc.format, desc()->input_scale_desc.format,
                                             desc()->input_shift_desc.format, desc()->output_scale_desc.format, desc()->output_shift_desc.format) &&
                       utils::one_of(desc()->src_desc.format, nc, tnc, nchw, nhwc, ncdhw, ndhwc, desired_blk_fmt) &&
                       utils::one_of(desc()->dst_desc.format, nc, tnc, nchw, nhwc, ncdhw, ndhwc, desired_blk_fmt)) &&
                       desc()->src_desc.format == desc()->dst_desc.format
        && IMPLICATION(desc()->alg_kind == mkldnn_binarization_depthwise,
                       utils::everyone_is(data_type::f32, desc()->src_desc.data_type) &&
                       utils::everyone_is(data_type::f32, desc()->thresholds_desc.data_type, desc()->output_mask_desc.data_type) &&
                       utils::everyone_is(data_type::bin, desc()->dst_desc.data_type) &&
                       utils::one_of(desc()->thresholds_desc.format, x) &&
                       utils::one_of(desc()->output_mask_desc.format, x) &&
                       utils::one_of(desc()->src_desc.format, nhwc) &&
                       utils::one_of(desc()->dst_desc.format, nhwc))
        && axis() == 1
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_quantization_fwd_t<isa>::jit_uni_quantization_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr) {
    const auto &desc = *pd()->desc();

    switch (desc.alg_kind) {
        case alg_kind::binarization_depthwise:
            kernel_ = new jit_uni_bin_depthwise_kernel<isa>(desc); break;
        case alg_kind::quantization_quantize_dequantize:
        case alg_kind::quantization_quantize:
            kernel_ = new jit_uni_quant_depthwise_kernel<isa>(desc);
        break;
        default: assert(!"unknown quantization alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_quantization_fwd_t<isa>::~jit_uni_quantization_fwd_t() {
    delete kernel_;
}

template struct jit_uni_quantization_fwd_t<sse42>;
template struct jit_uni_quantization_fwd_t<avx2>;
template struct jit_uni_quantization_fwd_t<avx512_common>;

}
}
}
