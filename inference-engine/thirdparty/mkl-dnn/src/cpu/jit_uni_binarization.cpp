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
#include "jit_uni_binarization.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

struct jit_args {
    const float* from;
    const uint8_t* to;
    const float* weights;
    const float* output_mask;
    size_t work_amount;
};

struct jit_uni_binarization_kernel_f32 : public c_compatible {
    const binarization_desc_t &desc_;
    void (*ker_)(const jit_args *);

    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_binarization_kernel_f32(const binarization_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_binarization_kernel_f32() {}
};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_bin_depthwise_kernel_f32 : public jit_uni_binarization_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bin_depthwise_kernel_f32)
    jit_uni_bin_depthwise_kernel_f32(const binarization_desc_t &desc)
        : jit_uni_binarization_kernel_f32(desc), jit_generator() {
        assert(one_of(desc.alg_kind, alg_kind::binarization_depthwise));
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_weights, ptr[param + GET_OFF(weights)]);
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
                uni_vmovups(vmm_wei(0), ptr[reg_weights + ch*step*sizeof(float)]);
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
            add(reg_weights, unrolled_loop_step*sizeof(float));
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
                uni_vmovups(vmm_wei(0), ptr[reg_weights + i*step*sizeof(float)]);
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
            add(reg_weights, main_loop_step*sizeof(float));
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
                    movss(xmm_wei(0), ptr[reg_weights + c * sizeof(float)]);
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
    Reg64 reg_weights = r11;
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

} /* namespace */

template <cpu_isa_t isa>
status_t jit_uni_binarization_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    auto desired_fmt = nhwc;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training, prop_kind::forward_inference)
        && utils::everyone_is(data_type::f32, desc()->src_desc.data_type, desc()->weights_desc.data_type,
                                              desc()->output_mask_desc.data_type)
        && utils::everyone_is(data_type::bin, desc()->dst_desc.data_type)
        && desc()->src_desc.format == desc()->dst_desc.format
        && utils::one_of(desc()->src_desc.format, desired_fmt)
        && utils::one_of(desc()->dst_desc.format, desired_fmt)
        && utils::one_of(desc()->weights_desc.format, x)
        && utils::one_of(desc()->output_mask_desc.format, x)
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_binarization_fwd_t<isa>::jit_uni_binarization_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
        case alg_kind::binarization_depthwise:
            kernel_ = new jit_uni_bin_depthwise_kernel_f32<isa>(desc); break;
        default: assert(!"unknown binarization alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_binarization_fwd_t<isa>::~jit_uni_binarization_fwd_t() {
    delete kernel_;
}

template <cpu_isa_t isa>
void jit_uni_binarization_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const src_data_t*>(this->input_memory(0));
    auto weights = reinterpret_cast<const src_data_t*>(this->input_memory(1));
    auto output_mask = reinterpret_cast<const src_data_t*>(this->input_memory(2));
    auto dst = reinterpret_cast<uint8_t*>(this->memory());

    const memory_desc_wrapper src_d(pd()->src_pd());
    const memory_desc_wrapper dst_d(pd()->dst_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper output_mask_d(pd()->weights_pd(1));

    const int N = src_d.dims()[0];
    const int C = src_d.dims()[1];
    const int H = src_d.dims()[2];
    const int W = src_d.dims()[3];

    int nbits = 8;

    parallel_nd(N, H, W,
        [&](int n, int h, int w) {
	auto arg = jit_args();

        arg.from    = &src[src_d.blk_off(n, 0, h, w)];
        arg.to      = &dst[dst_d.blk_off(n, 0, h, w) / nbits];
        arg.weights = &weights[weights_d.blk_off(0)];
        arg.output_mask = &output_mask[output_mask_d.blk_off(0)];
        arg.work_amount = (size_t)C;

        (*kernel_)(&arg);
    });
}

template struct jit_uni_binarization_fwd_t<sse42>;
template struct jit_uni_binarization_fwd_t<avx2>;
template struct jit_uni_binarization_fwd_t<avx512_common>;

}
}
}
