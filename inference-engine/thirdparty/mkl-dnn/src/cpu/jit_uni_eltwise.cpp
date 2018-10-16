/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <mkldnn_types.h>
#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

namespace {

template <cpu_isa_t isa>
struct jit_uni_relu_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_prepare_constants_f32)

    jit_uni_relu_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p, float alpha) {
        mov(p->imm_addr64, float2int(alpha));
        if (p->xmm_ns.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_ns, p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, p->xmm_ns);
        }

        uni_vpxor(p->vmm_zero, p->vmm_zero, p->vmm_zero);
    }
};

template <cpu_isa_t isa>
struct jit_uni_elu_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_elu_prepare_constants_f32)

    jit_uni_elu_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p, float alpha) {
        mov(p->imm_addr64, float2int(alpha));
        if (p->xmm_ns.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_ns, p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, p->xmm_ns);
        }

        uni_vpxor(p->vmm_zero, p->vmm_zero, p->vmm_zero);
    }
};

template <cpu_isa_t isa>
struct jit_uni_abs_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_abs_prepare_constants_f32)
    explicit jit_uni_abs_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p) {
        mov(p->imm_addr64, 0x7fffffff);
        if (p->xmm_aux0.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_aux0, p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, p->xmm_aux0);
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_sqrt_prepare_constants_f32 : jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_sqrt_prepare_constants_f32)
    explicit jit_uni_sqrt_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p) {
        uni_vpxor(p->vmm_zero, p->vmm_zero, p->vmm_zero);
    }
};

template <cpu_isa_t isa>
struct jit_uni_linear_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_linear_prepare_constants_f32)
    jit_uni_linear_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p, float alpha, float beta) {
        mov(p->imm_addr64, float2int(alpha));
        if (p->xmm_ns.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_ns, p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, p->xmm_ns);
        }

        mov(p->imm_addr64, float2int(beta));
        if (p->xmm_aux0.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_aux0, p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, p->xmm_aux0);
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_bounded_relu_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bounded_relu_prepare_constants_f32)
    jit_uni_bounded_relu_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p, float alpha) {
        mov(p->imm_addr64, float2int(alpha));
        if (p->xmm_ns.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_ns, p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, p->xmm_ns);
        }

        uni_vpxor(p->vmm_zero, p->vmm_zero, p->vmm_zero);
    }
};


template <cpu_isa_t isa>
struct jit_uni_clamp_prepare_constants_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_clamp_prepare_constants_f32)
    jit_uni_clamp_prepare_constants_f32(jit_uni_eltwise_vector_f32<isa>* p, float alpha, float beta) {
        mov(p->imm_addr64, float2int(alpha));
        if (p->xmm_ns.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_ns, p->imm_addr64);
            uni_vbroadcastss(p->vmm_ns, p->xmm_ns);
        }

        mov(p->imm_addr64, float2int(beta));
        if (p->xmm_aux0.getIdx() > 15) {
            uni_vmovups(p->vmm_src_rem, Vmm(0));
            movq(Xmm(0), p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, Xmm(0));
            uni_vmovups(Vmm(0), p->vmm_src_rem);
        } else {
            movq(p->xmm_aux0, p->imm_addr64);
            uni_vbroadcastss(p->vmm_aux0, p->xmm_aux0);
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_relu_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_compute_vector_f32)
    jit_uni_relu_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        unsigned char _cmp_gt_os = isa == avx512_common ? 14 : 6;

        uni_vmovups(p->vmm_src_rem, vmm_src);
        if (isa == sse42) {
            movups(p->vmm_mask, vmm_src);
            movups(vmm_dst, vmm_src);
            mulps(vmm_dst, p->vmm_ns);
            cmpps(p->vmm_mask, p->vmm_zero, _cmp_gt_os);
            blendvps(vmm_dst, p->vmm_src_rem);
        } else if (isa == avx2) {
            vmulps(vmm_dst, vmm_src, p->vmm_ns);
            vcmpgtps(p->vmm_mask, p->vmm_src_rem, p->vmm_zero);
            vblendvps(vmm_dst, vmm_dst, p->vmm_src_rem, p->vmm_mask);
        } else if (isa == avx512_common) {
            vmulps(vmm_dst, vmm_src, p->vmm_ns);
            vcmpps(p->k_mask, p->vmm_src_rem, p->vmm_zero, _cmp_gt_os);
            vblendmps(vmm_dst | p->k_mask, vmm_dst,
                      p->vmm_src_rem);
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_exp_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_exp_compute_vector_f32)
    jit_uni_exp_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src) {
        const unsigned char _op_floor = 1;

        uni_vminps(vmm_src, vmm_src, ptr[p->imm_addr64 + 10 * p->vlen]);
        uni_vmaxps(vmm_src, vmm_src, ptr[p->imm_addr64 + 11 * p->vlen]);
        uni_vmovups(p->vmm_aux0, vmm_src);
        //calculate exp(x)
        // fx = x * log2ef + 0.5
        uni_vmulps(vmm_src, vmm_src, ptr[p->imm_addr64 + 2 * p->vlen]);
        uni_vaddps(vmm_src, vmm_src, ptr[p->imm_addr64 + 1 * p->vlen]);

        // tmp = floorf(fx)
        if (isa == avx512_common) {
            vcvtps2dq(p->vmm_aux1 | T_rd_sae, vmm_src);
            vcvtdq2ps(p->vmm_aux1, p->vmm_aux1);

            unsigned char _cmp_gt_os = 14;
            Xbyak::Opmask k_mask_tmp = Xbyak::Opmask(2);
            vcmpps(k_mask_tmp, p->vmm_aux1, vmm_src, _cmp_gt_os);
            vmovups(p->vmm_aux2 | k_mask_tmp | T_z, zword[p->imm_addr64 + 0 * p->vlen]);

            uni_vsubps(p->vmm_aux1, p->vmm_aux1, p->vmm_aux2);
        } else {
            uni_vroundps(p->vmm_aux1, vmm_src, _op_floor);
        }

        //keep fx for further computations
        uni_vmovups(vmm_src, p->vmm_aux1); //vmm_src = fx

        //x = x - fx * ln2
        uni_vfnmadd231ps(p->vmm_aux0, p->vmm_aux1, ptr[p->imm_addr64 + 3 * p->vlen]);

        // compute 2^n
        uni_vcvtps2dq(p->vmm_aux1, vmm_src);
        uni_vpaddd(p->vmm_aux1, p->vmm_aux1, ptr[p->imm_addr64 + 4 * p->vlen]);
        uni_vpslld(p->vmm_aux1, p->vmm_aux1, 23); //Vmm(6) = 2^-fx

        // y = p5
        uni_vmovups(vmm_src, ptr[p->imm_addr64 + 9 * p->vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(vmm_src, p->vmm_aux0, ptr[p->imm_addr64 + 8 * p->vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(vmm_src, p->vmm_aux0, ptr[p->imm_addr64 + 7 * p->vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(vmm_src, p->vmm_aux0, ptr[p->imm_addr64 + 6 * p->vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(vmm_src, p->vmm_aux0, ptr[p->imm_addr64 + 0 * p->vlen]);
        // y = y * x + p0
        uni_vfmadd213ps(vmm_src, p->vmm_aux0, ptr[p->imm_addr64 + 5 * p->vlen]);  //exp(q)
        // y = y * 2^n
        uni_vmulps(vmm_src, vmm_src, p->vmm_aux1);
    }
};

template <cpu_isa_t isa>
struct jit_uni_elu_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_elu_compute_vector_f32)
    jit_uni_elu_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        const unsigned char _cmp_let_os = 2;
        const unsigned char _cmp_gt_os = 6;

        uni_vmovups(p->vmm_src_rem, vmm_src);
        uni_vmovups(vmm_dst, vmm_src);

        // compute exponent
        auto* generator = new jit_uni_exp_compute_vector_f32<isa>(p, vmm_dst);
        db(generator->getCode(), generator->getSize());

        // alpha * (exp(x) - 1)
        uni_vsubps(vmm_dst, vmm_dst, ptr[p->imm_addr64 + 0 * 32]);
        uni_vmulps(vmm_dst, vmm_dst, p->vmm_ns);

        // combine with mask
        if (isa == sse42) {
            pxor(p->vmm_mask, p->vmm_mask);
            cmpps(p->vmm_mask,  p->vmm_src_rem, _cmp_let_os);
            blendvps(vmm_dst, p->vmm_src_rem);
        } else if (isa == avx2) {
            uni_vpxor(p->vmm_zero, p->vmm_zero, p->vmm_zero);
            uni_vcmpgtps(p->vmm_mask, p->vmm_src_rem, p->vmm_zero);
            uni_vblendvps(vmm_dst, vmm_dst, p->vmm_src_rem, p->vmm_mask);
        } else if (isa == avx512_common) {
            vpxord(p->vmm_zero, p->vmm_zero, p->vmm_zero);
            vcmpps(p->k_mask, p->vmm_src_rem, p->vmm_zero, _cmp_gt_os);

            vblendmps(vmm_dst | p->k_mask, vmm_dst, p->vmm_src_rem);
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_tanh_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_tanh_compute_vector_f32)
    jit_uni_tanh_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute exp(2x)
        uni_vaddps(vmm_src, vmm_src, vmm_src);
        auto* generator = new jit_uni_exp_compute_vector_f32<isa>(p, vmm_src);
        db(generator->getCode(), generator->getSize());
        // dup exp(2x)
        uni_vmovups(p->vmm_aux0, vmm_src);
        // (exp(2x) - 1)
        uni_vsubps(vmm_src, vmm_src, ptr[p->imm_addr64 + 0 * 32]);
        // (exp(2x) + 1)
        uni_vaddps(p->vmm_aux0, p->vmm_aux0, ptr[p->imm_addr64 + 0 * 32]);
        // y = (exp(2x) - 1) / (exp(2x) + 1)
        uni_vdivps(vmm_dst, vmm_src, p->vmm_aux0);
    }
};

template <cpu_isa_t isa>
struct jit_uni_square_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_square_compute_vector_f32)
    jit_uni_square_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute exp(2x)
        uni_vmulps(vmm_dst, vmm_src, vmm_src);
    }
};

template <cpu_isa_t isa>
struct jit_uni_abs_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_abs_compute_vector_f32)
    jit_uni_abs_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute abs(x) = _mm_and_ps(x, 01111..111));
        uni_vandps(vmm_dst, vmm_src, p->vmm_aux0);
    }
};

template <cpu_isa_t isa>
struct jit_uni_sqrt_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_sqrt_compute_vector_f32)
    jit_uni_sqrt_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        unsigned char _cmp_gt_os = 6;

        if (isa ==avx512_common) {
            uni_vmovups(p->vmm_src_rem, vmm_src);

            vcmpps(p->k_mask, p->vmm_src_rem, p->vmm_zero, _cmp_gt_os);
            uni_vsqrtps(vmm_dst, vmm_src);

            vblendmps(vmm_dst | p->k_mask,  p->vmm_zero, vmm_dst);
        } else {
            uni_vmovups(p->vmm_src_rem, vmm_src);
            uni_vmovups(p->vmm_mask, vmm_src);
            uni_vmovups(vmm_dst, p->vmm_zero);
            uni_vcmpgtps(p->vmm_mask, p->vmm_mask, p->vmm_zero);

            // compute sqrt(x)
            uni_vsqrtps(p->vmm_src_rem, p->vmm_src_rem);

            // blend
            uni_vblendvps(vmm_dst, vmm_dst, p->vmm_src_rem, p->vmm_mask);
        }

    }
};

template <cpu_isa_t isa>
struct jit_uni_linear_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_linear_compute_vector_f32)
    jit_uni_linear_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute x = alpha * x + beta;
        uni_vfmadd213ps(vmm_src, p->vmm_ns, p->vmm_aux0);
        uni_vmovups(vmm_dst, vmm_src);
    }
};

template <cpu_isa_t isa>
struct jit_uni_bounded_relu_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_bounded_relu_compute_vector_f32)
    jit_uni_bounded_relu_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute bounded relu */
        uni_vmaxps(vmm_src, vmm_src, p->vmm_zero);
        uni_vminps(vmm_dst, vmm_src, p->vmm_ns);
    }
};

template <cpu_isa_t isa>
struct jit_uni_soft_relu_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_soft_relu_compute_vector_f32)
    jit_uni_soft_relu_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        const unsigned char _op_floor = 1;

        uni_vminps(vmm_src, vmm_src, ptr[p->imm_addr64 + 24 * p->vlen]);
        uni_vmaxps(vmm_src, vmm_src, ptr[p->imm_addr64 + 25 * p->vlen]);
        uni_vmovups(p->vmm_src_rem, vmm_src);
        // calculate exp(x)
        // fx = x * log2ef + 0.5
        uni_vmulps(vmm_src, vmm_src, ptr[p->imm_addr64 + 2 * p->vlen]);
        uni_vaddps(vmm_src, vmm_src, ptr[p->imm_addr64 + 1 * p->vlen]);

        // tmp = floorf(fx)
        if (isa == avx512_common) {
            vcvtps2dq(p->vmm_aux0 | T_rd_sae, vmm_src);
            vcvtdq2ps(p->vmm_aux0, p->vmm_aux0);

            unsigned char _cmp_gt_os = 14;
            Xbyak::Opmask k_mask_tmp = Xbyak::Opmask(2);
            vcmpps(k_mask_tmp, p->vmm_aux0, vmm_src, _cmp_gt_os);
            vmovups(p->vmm_aux2 | k_mask_tmp | T_z, zword[p->imm_addr64 + 0 * p->vlen]);

            uni_vsubps(p->vmm_aux0, p->vmm_aux0, p->vmm_aux2);
        } else {
            uni_vroundps(p->vmm_aux0, vmm_src, _op_floor);
        }

        // keep fx for further computations
        uni_vmovups(vmm_src, p->vmm_aux0); //Vmm(1) = fx
        // calculation fx * ln2
        uni_vmulps(p->vmm_aux0, p->vmm_aux0, ptr[p->imm_addr64 + 3 * p->vlen]);
        // x = x - fx * ln2
        uni_vsubps(p->vmm_src_rem, p->vmm_src_rem, p->vmm_aux0);
        // y = p5
        uni_vmovups(p->vmm_aux1, ptr[p->imm_addr64 + 22 * p->vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(p->vmm_aux1, p->vmm_src_rem, ptr[p->imm_addr64 + 21 * p->vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(p->vmm_aux1, p->vmm_src_rem, ptr[p->imm_addr64 + 20 * p->vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(p->vmm_aux1, p->vmm_src_rem, ptr[p->imm_addr64 + 19 * p->vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(p->vmm_aux1, p->vmm_src_rem, ptr[p->imm_addr64 + 0 * p->vlen]);
        // y = y * x + p0
        uni_vfmadd213ps(p->vmm_aux1, p->vmm_src_rem, ptr[p->imm_addr64 + 17 * p->vlen]);  //exp(q)

        // compute 2^(-n)
        if (isa == avx512_common) {
            uni_vmulps(p->vmm_src_rem, vmm_src, ptr[p->imm_addr64 + 23 * p->vlen]);
            uni_vcvtps2dq(p->vmm_src_rem, p->vmm_src_rem);
        } else {
            uni_vcvtps2dq(p->vmm_src_rem, vmm_src);
            uni_vpsignd(p->vmm_src_rem, p->vmm_src_rem, ptr[p->imm_addr64 + 23 * p->vlen]);
        }

        uni_vpaddd(p->vmm_src_rem, p->vmm_src_rem, ptr[p->imm_addr64 + 4 * p->vlen]);
        uni_vpslld(p->vmm_src_rem, p->vmm_src_rem, 23); //Vmm(6) = 2^-fx
        // calculate ln(1 + y)
        uni_vaddps(p->vmm_aux1, p->vmm_aux1, p->vmm_src_rem);
        // x = y; y is free; keep x for further computations
        uni_vmovups(vmm_src, p->vmm_aux1);
        // frexp()
        uni_vpsrld(vmm_src, vmm_src, 23);
        uni_vcvtdq2ps(vmm_src, vmm_src);
        // got n. where n is x = 2^n * y. y = 0.5 .. 1
        uni_vsubps(vmm_src, vmm_src, ptr[p->imm_addr64 + 5 * p->vlen]);

        uni_vandps(p->vmm_aux1, p->vmm_aux1, ptr[p->imm_addr64 + 6 * p->vlen]);
        // got y. (mantisa)  0.5 < y < 1
        uni_vorps(p->vmm_aux1, p->vmm_aux1, ptr[p->imm_addr64 + 7 * p->vlen]);
        // y  = y - 1
        uni_vsubps(p->vmm_aux1, p->vmm_aux1, ptr[p->imm_addr64 + 0 * p->vlen]);
        // y = p8
        uni_vmovups(p->vmm_src_rem, ptr[p->imm_addr64 + 16 * p->vlen]);
        // y = y * x + p7
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 15 * p->vlen]);
        // y = y * x + p6
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 14 * p->vlen]);
        // y = y * x + p5
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 13 * p->vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 12 * p->vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 11 * p->vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 10 * p->vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 9 * p->vlen]);
        // y = y * x + p0 ; p0 = 0
        uni_vfmadd213ps(p->vmm_src_rem, p->vmm_aux1, ptr[p->imm_addr64 + 8 * p->vlen]);
        //calculate ln(2) * n
        uni_vmulps(vmm_src, vmm_src, ptr[p->imm_addr64 + 3 * p->vlen]);
        uni_vaddps(p->vmm_src_rem, p->vmm_src_rem, vmm_src);
        uni_vaddps(p->vmm_src_rem, p->vmm_src_rem, p->vmm_aux0);

        uni_vmovups(vmm_dst, p->vmm_src_rem);
    }
};

template <cpu_isa_t isa>
struct jit_uni_logistic_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_logistic_compute_vector_f32)
    jit_uni_logistic_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        auto* generator = new jit_uni_exp_compute_vector_f32<isa>(p, vmm_src);
        db(generator->getCode(), generator->getSize());

        // dup exp(x)
        uni_vmovups(p->vmm_src_rem, vmm_src);
        // (exp(x) + 1)
        uni_vaddps(p->vmm_src_rem, p->vmm_src_rem, ptr[p->imm_addr64 + 0 * p->vlen]);
        // y = exp(x) / (exp(x) + 1)
        uni_vdivps(vmm_dst, vmm_src, p->vmm_src_rem);
    }
};

template <cpu_isa_t isa>
struct jit_uni_clamp_compute_vector_f32 : jit_generator {
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
            isa == avx2, Ymm, Zmm>::type;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_clamp_compute_vector_f32)
    jit_uni_clamp_compute_vector_f32(jit_uni_eltwise_vector_f32<isa>* p, const Vmm &vmm_src, const Vmm &vmm_dst) {
        // compute bounded relu */
        uni_vmaxps(vmm_src, vmm_src, p->vmm_aux0);
        uni_vminps(vmm_dst, vmm_src, p->vmm_ns);
    }
};

template <cpu_isa_t isa>
struct jit_uni_elu_prepare_table_f32 : jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_elu_prepare_table_f32)
    jit_uni_elu_prepare_table_f32(jit_uni_eltwise_vector_f32<isa>* p) {
        const unsigned int cvals[] = {
                0x3f800000, // [0] 1.0f
                0x3f000000, // [1] 0.5f
                0x3fb8aa3b, // [2] log2ef = 1.44269502f
                0x3f317218, // [3] ln2f =   0.69314718f
                0x0000007f, // [4] 0x7f
                // exp(x) polynom
                0x3f800001, // [5] p0 = 1.0000001f
                0x3efffe85, // [6] p2 = 0.4999887f
                0x3e2aaa3e, // [7] p3 = 0.16666505f
                0x3d2bb1b1, // [8] p4 = 0.041917507f
                0x3c091ec1, // [9] p5 = 0.008369149f
                0x42b0c0a5, //[10] max logf = 88.3762589f
                0xc1766666  //[11] min logf = -14.5f
        };

        for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
            for (size_t d = 0; d < p->vlen / sizeof(float); ++d) {
                dd(cvals[i]);
            }
        }
    }
};

template <cpu_isa_t isa>
struct jit_uni_soft_relu_prepare_table_f32 : jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_soft_relu_prepare_table_f32)
    jit_uni_soft_relu_prepare_table_f32(jit_uni_eltwise_vector_f32<isa>* p) {
        const unsigned int cvals[] = {
                0x3f800000, // [0] 1.0f
                0x3f000000, // [1] 0.5f
                0x3fb8aa3b, // [2] log2ef = 1.44269502f
                0x3f317218, // [3] ln2f =   0.69314718f
                0x0000007f, // [4] 0x7f
                0x42fc0000, // [5] 126
                0x807fffff, // [6] and with (to get 0.5 * mantissa)
                0x3f000000, // [7] or with (to get 0.5 * mantissa)
                // ln(1 + x) polynomial
                0xb2b4637d, // [8]  p0 = 0.0000000244f
                0x3f7fff8e, // [9]  p1 = 0.9999976971f
                0xbf001759, //[10]  p2 = -0.5002478215f
                0x3ea70608, //[11]  p3 = 0.3272714505f
                0xbea3d7bf, //[12]  p4 = -0.3153830071f
                0xbe361d04, //[13]  p5 = -0.1701777461f
                0xbfa8f1e6, //[14]  p6 = -1.3254635147f
                0xbfe1e812, //[15]  p7 = -1.7971917960f
                0xbfc4d30e, //[16]  p8 = -1.5652673123f
                // exp(x) polynomial
                0x3f800001, //[17]  p0 = 1.0000001f
                0x3f800000, //[18]  p1 = 1.0f
                0x3efffe85, //[19]  p2 = 0.4999887f
                0x3e2aaa3e, //[20]  p3 = 0.16666505f
                0x3d2bb1b1, //[21]  p4 = 0.041917507f
                0x3c091ec1, //[22]  p5 = 0.008369149f
                0xbf800000, //[23] is required for sign changing
                0x42b0c0a5, //[24] max logf = 88.3762589f
                0xc1766666  //[25] min logf = -14.5f
        };

        for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
            for (size_t d = 0; d < p->vlen / sizeof(float); ++d) {
                dd(cvals[i]);
            }
        }
    }
};

} // namespace

template <cpu_isa_t isa>
void jit_uni_eltwise_vector_f32<isa>::init(alg_kind_t elt_alg_, nstl::vector<int> &shared_vecs, nstl::vector<Reg64> &shared_regs) {
    assert(utils::one_of(elt_alg_, alg_kind::eltwise_relu, alg_kind::eltwise_tanh, alg_kind::eltwise_elu,
                         alg_kind::eltwise_square, alg_kind::eltwise_abs, alg_kind::eltwise_sqrt, alg_kind::eltwise_linear,
                         alg_kind::eltwise_bounded_relu, alg_kind::eltwise_soft_relu, alg_kind::eltwise_logistic,
                         alg_kind::eltwise_clamp));

    assert(isa == sse42 || isa == avx2 || isa == avx512_common);
    // TODO (dmitrygo): for isa == sse42 mask have to be Xmm(0). Need to check this.
    vmm_mask = Vmm(shared_vecs[0]);
    vmm_src_rem = Vmm(shared_vecs[3]);
    vmm_ns = Vmm(shared_vecs[1]);
    xmm_ns = Xmm(shared_vecs[1]);
    vmm_zero = Vmm(shared_vecs[2]);
    vmm_aux0 = Vmm(shared_vecs[0]);
    xmm_aux0 = Xmm(shared_vecs[0]);
    vmm_aux1 = Vmm(shared_vecs[2]);
    if (isa == avx512_common)
        vmm_aux2 = Vmm(shared_vecs[4]);

    imm_addr64 = shared_regs[0];

    // TODO (dmitrygo): we should share opmasks either?
    k_mask = Opmask(1);

    elt_alg = elt_alg_;
}

template <cpu_isa_t isa>
jit_code_injection jit_uni_eltwise_vector_f32<isa>::prepareConstants(float alpha, float beta) {
    if (generator != nullptr) { delete generator; generator = nullptr; }

    switch(elt_alg) {
        case alg_kind::eltwise_relu: {
            generator = new jit_uni_relu_prepare_constants_f32<isa>(this, alpha);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_elu: {
            generator = new jit_uni_elu_prepare_constants_f32<isa>(this, alpha);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_abs: {
            generator = new jit_uni_abs_prepare_constants_f32<isa>(this);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_sqrt: {
            generator = new jit_uni_sqrt_prepare_constants_f32<isa>(this);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_linear: {
            generator = new jit_uni_linear_prepare_constants_f32<isa>(this, alpha, beta);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_bounded_relu: {
            generator = new jit_uni_bounded_relu_prepare_constants_f32<isa>(this, alpha);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_clamp: {
            generator = new jit_uni_clamp_prepare_constants_f32<isa>(this, alpha, beta);
            return {generator->getCode(), generator->getSize()};
        }
        default: {
            return {};
        }
    }
}

template <cpu_isa_t isa>
jit_code_injection jit_uni_eltwise_vector_f32<isa>::computeVector(const Vmm &vmm_src, const Vmm &vmm_dst) {
    if (generator != nullptr) { delete generator; generator = nullptr; }

    switch(elt_alg) {
        case alg_kind::eltwise_relu: {
            generator = new jit_uni_relu_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_elu: {
            generator = new jit_uni_elu_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_tanh: {
            generator = new jit_uni_tanh_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_square: {
            generator = new jit_uni_square_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_abs: {
            generator = new jit_uni_abs_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_sqrt: {
            generator = new jit_uni_sqrt_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_linear: {
            generator = new jit_uni_linear_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_bounded_relu: {
            generator = new jit_uni_bounded_relu_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_soft_relu: {
            generator = new jit_uni_soft_relu_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_logistic: {
            generator = new jit_uni_logistic_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_clamp: {
            generator = new jit_uni_clamp_compute_vector_f32<isa>(this, vmm_src, vmm_dst);
            return {generator->getCode(), generator->getSize()};
        }
        default: {
            return {};
        }
    }
}

template <cpu_isa_t isa>
jit_code_injection jit_uni_eltwise_vector_f32<isa>::prepareTable() {
    if (generator != nullptr) { delete generator; generator = nullptr; }

    switch(elt_alg) {
        case alg_kind::eltwise_elu:
        case alg_kind::eltwise_tanh:
        case alg_kind::eltwise_logistic: {
            generator = new jit_uni_elu_prepare_table_f32<isa>(this);
            return {generator->getCode(), generator->getSize()};
        }
        case alg_kind::eltwise_soft_relu: {
            generator = new jit_uni_soft_relu_prepare_table_f32<isa>(this);
            return {generator->getCode(), generator->getSize()};
        }
        default: {
            return {};
        }
    }
}

template struct jit_uni_eltwise_vector_f32<avx512_common>;
template struct jit_uni_eltwise_vector_f32<avx2>;
template struct jit_uni_eltwise_vector_f32<sse42>;


struct jit_args {
    const float *from;
    const float *for_comparison;
    const float *to;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_f32 : public c_compatible {
    const eltwise_desc_t &desc_;

    void (*ker_)(const jit_args *);
    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_eltwise_kernel_f32(const eltwise_desc_t &desc)
        : desc_(desc), ker_(nullptr) {}
    virtual ~jit_uni_eltwise_kernel_f32() {}

protected:
    bool is_bwd() const { return desc_.prop_kind == prop_kind::backward_data; }
};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_relu_kernel_f32 : public jit_uni_eltwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_relu_kernel_f32)

    void compute_step(bool vectorize, const int uf, const int shift) {
        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(Vmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    uni_vmovups(Vmm(uf + i + 1),
                                ptr[reg_for_comparison + i * shift]);
            } else {
                movss(Xmm(i + 1), ptr[reg_from + i * shift]);
                if (is_bwd())
                    movss(Xmm(uf + i + 1),
                          ptr[reg_for_comparison + i * shift]);
            }
        }

        if (isa == sse42) {
            for (int i = 0; i < uf; i++) {
                movups(Vmm(2 * uf + i + 1), Vmm(i + 1));
                mulps(Vmm(2 * uf + i + 1), vmm_ns);

                Vmm mask = Vmm(0);
                if (is_bwd()) {
                    movups(mask, Vmm(uf + i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                } else {
                    movups(mask, Vmm(i + 1));
                    cmpps(mask, vmm_zero, _cmp_nle_us);
                }
                blendvps(Vmm(2 * uf + i + 1), Vmm(i + 1));
            }
        } else {
            for (int i = 0; i < uf; i++) {
                vmulps(Vmm(2 * uf + i + 1), Vmm(i + 1), vmm_ns);
                if (isa == avx2) {
                    if (is_bwd())
                        vcmpgtps(vmm_mask, Vmm(uf + i + 1), vmm_zero);
                    else
                        vcmpgtps(vmm_mask, Vmm(i + 1), vmm_zero);

                    vblendvps(Vmm(2 * uf + i + 1), Vmm(2 * uf + i + 1),
                              Vmm(i + 1), vmm_mask);

                } else {
                    if (is_bwd())
                        vcmpps(k_mask, Vmm(uf + i + 1), vmm_zero, _cmp_nle_us);
                    else
                        vcmpps(k_mask, Vmm(i + 1), vmm_zero, _cmp_nle_us);
                    vblendmps(Vmm(2 * uf + i + 1) | k_mask, Vmm(2 * uf + i + 1),
                              Vmm(i + 1));
                }
            }
        }

        for (int i = 0; i < uf; i++) {
            if (vectorize) {
                uni_vmovups(ptr[reg_to + i * shift], Vmm(2 * uf + i + 1));
            } else {
                movss(ptr[reg_to + i * shift], Xmm(2 * uf + i + 1));
            }
        }
    }

    jit_uni_relu_kernel_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator() {
        assert(desc.alg_kind == alg_kind::eltwise_relu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        Reg64 param = abi_param1;

        const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
        const int loop_dec[] = {simd_w, 1};
        const int uf[] = {1, 1};
        const int shift[] = {cpu_isa_traits<isa>::vlen, sizeof(float)};
        const bool loop_vectorize[] = {true, false};

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        if (is_bwd())
            mov(reg_for_comparison, ptr[param + GET_OFF(for_comparison)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        mov(imm_addr64, float2int(desc.alpha));
        movq(xmm_ns, imm_addr64);
        uni_vbroadcastss(vmm_ns, xmm_ns);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        Label loop_label[3];

        for (int id = 0; id < 2; id++) {
            L(loop_label[id]);
            cmp(reg_work_amount, uf[id] * loop_dec[id] - 1);
            jle(loop_label[id + 1], T_NEAR);

            compute_step(loop_vectorize[id], uf[id], shift[id]);

            add(reg_from, uf[id] * shift[id]);
            add(reg_to, uf[id] * shift[id]);
            if (is_bwd())
                add(reg_for_comparison, uf[id] * shift[id]);

            sub(reg_work_amount, uf[id] * loop_dec[id]);
            jmp(loop_label[id]);
        }

        L(loop_label[2]);
        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    Reg64 reg_from = rax;
    Reg64 reg_for_comparison = is_bwd() ? rdx : reg_from;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;

    Xmm xmm_ns = Xmm(14);

    Vmm vmm_ns = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmm_zero = Vmm(isa == avx512_common ? 31 : 15);

    Vmm vmm_mask = Vmm(isa == avx512_common ? 28 : 12);
    Opmask k_mask = Opmask(1);
};

template <cpu_isa_t isa>
struct jit_uni_kernel_fwd_f32: public jit_uni_eltwise_kernel_f32,
    public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel_fwd_f32)

    jit_uni_kernel_fwd_f32(const eltwise_desc_t &desc)
        : jit_uni_eltwise_kernel_f32(desc), jit_generator() {
        using namespace alg_kind;

        assert(is_bwd() == false);
        assert(utils::one_of(desc.alg_kind, eltwise_tanh, eltwise_elu,
                    eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                    eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                    eltwise_clamp));

        typedef void (jit_uni_kernel_fwd_f32<isa>::*func_t)();
        func_t prepare_const, vectorized_body, reminder_body, prepare_table;

        switch(desc.alg_kind) {
        case eltwise_tanh:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::tanh_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::tanh_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_table;
            break;
        case eltwise_elu:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::elu_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::elu_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::elu_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_table;
            break;
        case eltwise_square:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::not_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::square_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::square_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        case eltwise_abs:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::abs_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::abs_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::abs_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        case eltwise_sqrt:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::sqrt_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::sqrt_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::sqrt_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        case eltwise_linear:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::linear_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::linear_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::linear_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        case eltwise_bounded_relu:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::bounded_relu_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::bounded_relu_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::bounded_relu_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        case eltwise_soft_relu:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::soft_relu_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::soft_relu_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::soft_relu_prepare_table;
            break;
        case eltwise_logistic:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::logistic_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::logistic_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::exp_prepare_table;
            break;
        case eltwise_clamp:
            prepare_const = &jit_uni_kernel_fwd_f32<isa>::clamp_prepare_const;
            vectorized_body = &jit_uni_kernel_fwd_f32<isa>::clamp_vectorized_body;
            reminder_body = &jit_uni_kernel_fwd_f32<isa>::clamp_reminder_body;
            prepare_table = &jit_uni_kernel_fwd_f32<isa>::not_prepare_table;
            break;
        default:
            assert(!"unknown eltwise alg_kind");
            prepare_const = NULL;
            vectorized_body = NULL;
            reminder_body = NULL;
            prepare_table = NULL;
            // XXX: handle this case better....
        }

        preamble();

        Reg64 param = abi_param1;
        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        assert(prepare_const);
        (this->*prepare_const)();

        cmp(reg_work_amount, simd_w);
        jl("reminder_loop_start", T_NEAR);

        L("vectorized_loop_start");

        assert(vectorized_body);
        (this->*vectorized_body)();

        add(reg_from, vlen);
        add(reg_to, vlen);

        sub(reg_work_amount, simd_w);
        cmp(reg_work_amount, simd_w);
        jge("vectorized_loop_start", T_NEAR);

        L("vectorized_loop_end");

        L("reminder_loop_start");

        cmp(reg_work_amount, 0);
        jle("reminder_loop_end", T_NEAR);

        assert(reminder_body);
        (this->*reminder_body)();

        add(reg_from, sizeof(float));
        add(reg_to, sizeof(float));

        dec(reg_work_amount);
        jmp("reminder_loop_start", T_NEAR);

        L("reminder_loop_end");

        postamble();

        // prepare consts for exp calculation
        assert(prepare_table);
        (this->*prepare_table)();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                isa == avx2, Ymm, Zmm>::type;

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int vlen   = cpu_isa_traits<isa>::vlen;

    unsigned char _op_floor = 1;

    Reg64 reg_from = rax;
    Reg64 reg_to = r8;
    Reg64 reg_work_amount = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 reg_mask = r9;

    Opmask k_mask = Opmask(1);
    Opmask k_mask_tmp = Opmask(2);

    Xmm xmm_mask = Xmm(0);
    Vmm vmm_mask = Vmm(0);

    Xmm xmm_src = Xmm(1);
    Vmm vmm_src = Vmm(1);

    Xmm xmm_dst = Xmm(2);
    Vmm vmm_dst = Vmm(2);

    Vmm vmm_tmp2 = Vmm(12);

    Xmm xmm_alpha = Xmm(13);
    Vmm vmm_alpha = Vmm(13);
    Xmm xmm_beta  = Xmm(14);
    Vmm vmm_beta  = Vmm(14);

    Xmm xmm_one = Xmm(11);
    Vmm vmm_one = Vmm(11);

    Xmm xmm_zero = Xmm(15);
    Vmm vmm_zero = Vmm(15);

    Label l_table;

    void not_prepare_table() {}

    void not_prepare_const() {}

    void exp_prepare_table() {
        const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            // exp(x) polynomial
            0x3f800001, // [5] p0 = 1.0000001f
            0x3efffe85, // [6] p2 = 0.4999887f
            0x3e2aaa3e, // [7] p3 = 0.16666505f
            0x3d2bb1b1, // [8] p4 = 0.041917507f
            0x3c091ec1, // [9] p5 = 0.008369149f
            0x42b0c0a5, //[10] max logf = 88.3762589f
            0xc1766666  //[11] min logf = -14.5f
        };

        align(64);
        L(l_table);
        for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(cvals[i]);
            }
        }
    }

    void exp_prepare_const() {
        // required for exp calculation
        mov(imm_addr64, l_table);
        uni_vmovups(vmm_one, ptr[imm_addr64 + 0 * vlen]);
    }

    void exp_vectorized() {
        uni_vminps(vmm_src, vmm_src, ptr[imm_addr64 + 10 * vlen]);
        uni_vmaxps(vmm_src, vmm_src, ptr[imm_addr64 + 11 * vlen]);
        uni_vmovups(Vmm(8), vmm_src);
        // calculate exp(x)
        // fx = x * log2ef + 0.5
        uni_vmulps(vmm_src, vmm_src, ptr[imm_addr64 + 2 * vlen]);
        uni_vaddps(vmm_src, vmm_src, ptr[imm_addr64 + 1 * vlen]);

        // tmp = floorf(fx)
        if (isa < avx512_common) {
            uni_vroundps(Vmm(5), vmm_src, _op_floor);
        } else {
            vcvtps2dq(Vmm(5) | T_rd_sae, vmm_src);
            vcvtdq2ps(Vmm(5), Vmm(5));

            vcmpps(k_mask_tmp, Vmm(5), vmm_src, _cmp_nle_us);
            vmovups(vmm_tmp2 | k_mask_tmp | T_z, zword[imm_addr64 + 0 * vlen]);

            // fx = fx - 1 (if there are fraction bits)
            uni_vsubps(Vmm(5), Vmm(5), vmm_tmp2);
        }
        // keep fx for further computations
        uni_vmovups(vmm_src, Vmm(5)); //vmm_src = fx
        // x = x - fx * ln2
        uni_vfnmadd231ps(Vmm(8), Vmm(5), ptr[imm_addr64 + 3 * vlen]);
        // y = p5
        uni_vmovups(vmm_dst, ptr[imm_addr64 + 9 * vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(vmm_dst, Vmm(8), ptr[imm_addr64 + 8 * vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(vmm_dst, Vmm(8), ptr[imm_addr64 + 7 * vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(vmm_dst, Vmm(8), ptr[imm_addr64 + 6 * vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(vmm_dst, Vmm(8), vmm_one);
        // y = y * x + p0
        uni_vfmadd213ps(vmm_dst, Vmm(8), ptr[imm_addr64 + 5 * vlen]);  //exp(q)
        // compute 2^n
        uni_vcvtps2dq(Vmm(6), vmm_src);
        uni_vpaddd(Vmm(6), Vmm(6), ptr[imm_addr64 + 4 * vlen]);
        uni_vpslld(Vmm(6), Vmm(6), 23); //Vmm(6) = 2^-fx
        // y = y * 2^n
        uni_vmulps(vmm_dst, vmm_dst, Vmm(6));
    }

    void exp_scalar() {
        minss(xmm_src, ptr[imm_addr64 + 10 * vlen]);
        maxss(xmm_src, ptr[imm_addr64 + 11 * vlen]);
        movups(Xmm(8), xmm_src);
        //calculate exp(x)
        // fx = x * log2ef + 0.5
        mulss(xmm_src, ptr[imm_addr64 + 2 * vlen]);
        addss(xmm_src, ptr[imm_addr64 + 1 * vlen]);
        // tmp = floorf(fx)
        roundss(Xmm(5), xmm_src, _op_floor);
        //keep fx for further computations
        movups(xmm_src, Xmm(5)); //xmm_src = fx
        //calculation fx * ln2
        mulss(Xmm(5), ptr[imm_addr64 + 3 * vlen]);
        //x = x - fx * ln2
        subss(Xmm(8), Xmm(5));
        // y = p5
        movups(xmm_dst, ptr[imm_addr64 + 9 * vlen]);
        // y = y * x + p4
        mulss(xmm_dst, Xmm(8));
        addss(xmm_dst, ptr[imm_addr64 + 8 * vlen]);

        // y = y * x + p3
        mulss(xmm_dst, Xmm(8));
        addss(xmm_dst, ptr[imm_addr64 + 7 * vlen]);
        // y = y * x + p2
        mulss(xmm_dst, Xmm(8));
        addss(xmm_dst, ptr[imm_addr64 + 6 * vlen]);

        // y = y * x + p1
        mulss(xmm_dst, Xmm(8));
        addss(xmm_dst, xmm_one);

        // y = y * x + p0
        mulss(xmm_dst, Xmm(8));
        addss(xmm_dst, ptr[imm_addr64 + 5 * vlen]); //exp(q)
        // compute 2^n
        cvtps2dq(Xmm(6), xmm_src);
        paddd(Xmm(6), ptr[imm_addr64 + 4 * vlen]);
        pslld(Xmm(6), 23); //Xmm(6) = 2^-fx
        // y = y * 2^n
        mulps(xmm_dst, Xmm(6));
    }

    void tanh_vectorized_body() {
        uni_vmovups(vmm_src, ptr[reg_from]);

        // compute exp(2x)
        uni_vaddps(vmm_src, vmm_src, vmm_src);
        exp_vectorized();

        // dup exp(2x)
        uni_vmovups(Vmm(14), vmm_dst);
        // (exp(2x) - 1)
        uni_vsubps(vmm_dst, vmm_dst, vmm_one);
        // (exp(2x) + 1)
        uni_vaddps(Vmm(14), Vmm(14), vmm_one);
        // y = (exp(2x) - 1) / (exp(2x) + 1)
        uni_vdivps(vmm_dst, vmm_dst, Vmm(14));

        // store result
        uni_vmovups(ptr[reg_to], vmm_dst);
    }

    void tanh_reminder_body() {
        movss(xmm_src, ptr[reg_from]);

        // compute exp(2x)
        addps(xmm_src, xmm_src);
        exp_scalar();

        movaps(Xmm(14), xmm_dst);
        // (exp(2x) - 1)
        subss(xmm_dst, xmm_one);
        // (exp(2x) + 1)
        addss(Xmm(14), ptr[imm_addr64 + 0 * vlen]);
        // y = (exp(2x) - 1) / (exp(2x) + 1)
        divss(xmm_dst, Xmm(14));

        // store result
        movss(ptr[reg_to], xmm_dst);
    }

    void elu_prepare_const() {
        mov(imm_addr64, float2int(desc_.alpha));
        movq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        //need for exp calculation
        mov(imm_addr64, l_table);
        uni_vmovups(vmm_one, ptr[imm_addr64 + 0 * vlen]);
    }

    void elu_vectorized_body() {
        uni_vmovups(vmm_src, ptr[reg_from]);
        // compute mask
        if (isa < avx512_common) {
            uni_vmovups(vmm_mask, vmm_src);
            uni_vcmpgtps(vmm_mask, vmm_mask, vmm_zero);
            // early exit if all elems positive
            uni_vmovmskps(reg_mask, vmm_mask);
        } else {
            vcmpps(k_mask, vmm_src, vmm_zero, _cmp_nle_us);
            kmovw(reg_mask.cvt32(), k_mask);
        }
        cmp(reg_mask, isa == sse42 ? 0x0f : (isa == avx2 ? 0xff : 0xffff));
        je("early_exit", T_NEAR);

        // compute exponent
        uni_vmovups(Vmm(10), vmm_src);
        exp_vectorized();

        // alpha * (exp(x) - 1)
        uni_vsubps(vmm_dst, vmm_dst, vmm_one);
        uni_vmulps(vmm_dst, vmm_dst, vmm_alpha);
        // combine with mask
        if (isa < avx512_common)
            uni_vblendvps(vmm_dst, vmm_dst, Vmm(10), vmm_mask);
        else
            vblendmps(vmm_dst | k_mask, vmm_dst, Vmm(10));
        // store result
        uni_vmovups(ptr[reg_to], vmm_dst);
        jmp("exit", T_NEAR);

        L("early_exit");
        uni_vmovups(ptr[reg_to], vmm_src);

        L("exit");
    }

    void elu_reminder_body() {
        movss(xmm_src, ptr[reg_from]);
        // compute mask
        movss(xmm_mask, xmm_src);
        cmpss(xmm_mask, xmm_zero, _cmp_nle_us);

        // early exit if all elems positive
        movmskps(reg_mask, xmm_mask);
        cmp(reg_mask, 0x01);
        je("reminder_early_exit", T_NEAR);

        // compute exponent
        movss(Xmm(10), xmm_src);
        exp_scalar();
        // alpha * (exp(x) - 1)
        subss(xmm_dst, xmm_one);
        mulss(xmm_dst, xmm_alpha);
        // combine with mask (in xmm0)
        blendvps(xmm_dst, Xmm(10));
        // store result
        movss(ptr[reg_to], xmm_dst);
        jmp("reminder_exit", T_NEAR);

        L("reminder_early_exit");
        movss(ptr[reg_to], xmm_src);

        L("reminder_exit");
    }

    void square_vectorized_body() {
        //load src
        uni_vmovups(vmm_src, ptr[reg_from]);

        // compute x*x
        uni_vmulps(vmm_src, vmm_src, vmm_src);

        // store result
        uni_vmovups(ptr[reg_to], vmm_src);
    }

    void square_reminder_body() {
        //load src
        movss(xmm_src, ptr[reg_from]);

        // compute x*x
        mulss(xmm_src, xmm_src);

        // store result
        movss(ptr[reg_to], xmm_src);
    }

    void abs_prepare_const() {
        mov(imm_addr64, 0x7fffffff);
        movq(xmm_one, imm_addr64);
        uni_vbroadcastss(vmm_one, xmm_one);
    }

    void abs_vectorized_body() {
        //load src
        uni_vmovups(vmm_src, ptr[reg_from]);

        // compute abs(x) = _mm_and_ps(x, 01111..111));
        uni_vandps(vmm_src, vmm_src, vmm_one);

        // store result
        uni_vmovups(ptr[reg_to], vmm_src);
    }

    void abs_reminder_body() {
        //load src
        movss(xmm_src, ptr[reg_from]);

        // compute abs(x)
        andps(xmm_src, xmm_one);

        // store result
        movss(ptr[reg_to], xmm_src);
    }

    void sqrt_prepare_const() {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    void sqrt_vectorized_body() {
        //load src
        uni_vmovups(vmm_src, ptr[reg_from]);

        uni_vmovups(vmm_mask, vmm_src);
        uni_vmovups(vmm_dst, vmm_zero);
        uni_vcmpgtps(vmm_mask, vmm_mask, vmm_zero);

        // early exit if all elems are negative
        uni_vmovmskps(reg_mask, vmm_mask);
        cmp(reg_mask, 0);
        je("early_exit", T_NEAR);

        // compute sqrt(x)
        uni_vsqrtps(vmm_src, vmm_src);

        // blend
        uni_vblendvps(vmm_dst, vmm_dst, vmm_src, vmm_mask);

        // store result
        L("early_exit");
        uni_vmovups(ptr[reg_to], vmm_dst);
    }

    void sqrt_reminder_body() {
        // load src
        movss(xmm_src, ptr[reg_from]);

        // compute mask
        movss(xmm_mask, xmm_src);
        movss(xmm_dst, xmm_zero);
        cmpss(xmm_mask, xmm_zero, _cmp_nle_us);

        // early exit if all elements are negative
        movmskps(reg_mask, xmm_mask);
        cmp(reg_mask, 0);
        je("reminder_early_exit", T_NEAR);

        // compute sqrt(x)
        sqrtss(xmm_src, xmm_src);

        // blend
        blendvps(xmm_dst, xmm_src);

        // store result
        L("reminder_early_exit");
        movss(ptr[reg_to], xmm_dst);
    }

    void linear_prepare_const() {
        mov(imm_addr64, float2int(desc_.alpha));
        movq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        mov(imm_addr64, float2int(desc_.beta));
        movq(xmm_beta, imm_addr64);
        uni_vbroadcastss(vmm_beta, xmm_beta);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    void linear_vectorized_body() {
        // load src
        uni_vmovups(vmm_src, ptr[reg_from]);

        // compute x = alpha * x + beta;
        uni_vfmadd213ps(vmm_src, vmm_alpha, vmm_beta);

        // store result
        uni_vmovups(ptr[reg_to], vmm_src);
    }

    void linear_reminder_body() {
        // load src
        movss(xmm_src, ptr[reg_from]);

        // compute x = alpha * x + beta;
        mulss(xmm_src, xmm_alpha);
        addss(xmm_src, xmm_beta);

        // store result
        movss(ptr[reg_to], xmm_src);
    }

    void bounded_relu_prepare_const() {
        mov(imm_addr64, float2int(desc_.alpha));
        movq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
    }

    void bounded_relu_vectorized_body() {
        uni_vmovups(vmm_src, ptr[reg_from]);
        // compute bounded relu */
        uni_vmaxps(vmm_src, vmm_src, vmm_zero);
        uni_vminps(vmm_src, vmm_src, vmm_alpha);
        // store result
        uni_vmovups(ptr[reg_to], vmm_src);
    }

    void bounded_relu_reminder_body() {
        movss(xmm_src, ptr[reg_from]);
        // compute bounded relu */
        maxps(xmm_src, xmm_zero);
        minps(xmm_src, xmm_alpha);
        // store result
        movss(ptr[reg_to], xmm_src);
    }

    void soft_relu_prepare_table() {
        const unsigned int cvals[] = {
            0x3f800000, // [0] 1.0f
            0x3f000000, // [1] 0.5f
            0x3fb8aa3b, // [2] log2ef = 1.44269502f
            0x3f317218, // [3] ln2f =   0.69314718f
            0x0000007f, // [4] 0x7f
            0x42fc0000, // [5] 126
            0x807fffff, // [6] and with (to get 0.5 * mantissa)
            0x3f000000, // [7] or with (to get 0.5 * mantissa)
            // ln(1 + x) polynomial
            0xb2b4637d, // [8]  p0 = 0.0000000244f
            0x3f7fff8e, // [9]  p1 = 0.9999976971f
            0xbf001759, //[10]  p2 = -0.5002478215f
            0x3ea70608, //[11]  p3 = 0.3272714505f
            0xbea3d7bf, //[12]  p4 = -0.3153830071f
            0xbe361d04, //[13]  p5 = -0.1701777461f
            0xbfa8f1e6, //[14]  p6 = -1.3254635147f
            0xbfe1e812, //[15]  p7 = -1.7971917960f
            0xbfc4d30e, //[16]  p8 = -1.5652673123f
            // exp(x) polynomial
            0x3f800001, //[17]  p0 = 1.0000001f
            0x3f800000, //[18]  p1 = 1.0f
            0x3efffe85, //[19]  p2 = 0.4999887f
            0x3e2aaa3e, //[20]  p3 = 0.16666505f
            0x3d2bb1b1, //[21]  p4 = 0.041917507f
            0x3c091ec1, //[22]  p5 = 0.008369149f
            0xbf800000, //[23] is required for sign changing
            0x42b0c0a5, //[24] max logf = 88.3762589f
            0xc1766666  //[25] min logf = -14.5f
        };

        align(64);
        L(l_table);
        for (size_t i = 0; i < sizeof(cvals) / sizeof(cvals[0]); ++i) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(cvals[i]);
            }
        }
    }

    void soft_relu_vectorized() {
        uni_vminps(Vmm(1), Vmm(1), ptr[imm_addr64 + 24 * vlen]);
        uni_vmaxps(Vmm(1), Vmm(1), ptr[imm_addr64 + 25 * vlen]);
        uni_vmovups(Vmm(8), Vmm(1));
        // calculate exp(x)
        // fx = x * log2ef + 0.5
        uni_vmulps(Vmm(1), Vmm(1), ptr[imm_addr64 + 2 * vlen]);
        uni_vaddps(Vmm(1), Vmm(1), ptr[imm_addr64 + 1 * vlen]);
        // tmp = floorf(fx)
        uni_vroundps(Vmm(5), Vmm(1), _op_floor);
        // keep fx for further computations
        uni_vmovups(Vmm(1), Vmm(5)); //Vmm(1) = fx
        // calculation fx * ln2
        uni_vmulps(Vmm(5), Vmm(5), ptr[imm_addr64 + 3 * vlen]);
        // x = x - fx * ln2
        uni_vsubps(Vmm(8), Vmm(8), Vmm(5));
        // y = p5
        uni_vmovups(Vmm(3), ptr[imm_addr64 + 22 * vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(Vmm(3), Vmm(8), ptr[imm_addr64 + 21 * vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(Vmm(3), Vmm(8), ptr[imm_addr64 + 20 * vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(Vmm(3), Vmm(8), ptr[imm_addr64 + 19 * vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(Vmm(3), Vmm(8), vmm_one);
        // y = y * x + p0
        uni_vfmadd213ps(Vmm(3), Vmm(8), ptr[imm_addr64 + 17 * vlen]);  //exp(q)
        // compute 2^(-n)
        uni_vcvtps2dq(Vmm(6), Vmm(1));
        uni_vpsignd(Vmm(6),Vmm(6), ptr[imm_addr64 + 23 * vlen]);
        uni_vpaddd(Vmm(6), Vmm(6), ptr[imm_addr64 + 4 * vlen]);
        uni_vpslld(Vmm(6), Vmm(6), 23); //Vmm(6) = 2^-fx
        // calculate ln(1 + y)
        uni_vaddps(Vmm(3), Vmm(3), Vmm(6));
        // x = y; y is free; keep x for further computations
        uni_vmovups(Vmm(1), Vmm(3));
        // frexp()
        uni_vpsrld(Vmm(1), Vmm(1), 23);
        uni_vcvtdq2ps(Vmm(1), Vmm(1));
        // got n. where n is x = 2^n * y. y = 0.5 .. 1
        uni_vsubps(Vmm(1), Vmm(1), ptr[imm_addr64 + 5 * vlen]);

        uni_vandps(Vmm(3), Vmm(3), ptr[imm_addr64 + 6 * vlen]);
        // got y. (mantisa)  0.5 < y < 1
        uni_vorps(Vmm(3), Vmm(3), ptr[imm_addr64 + 7 * vlen]);
        // y  = y - 1
        uni_vsubps(Vmm(3), Vmm(3), vmm_one);
        // y = p8
        uni_vmovups(Vmm(8), ptr[imm_addr64 + 16 * vlen]);
        // y = y * x + p7
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 15 * vlen]);
        // y = y * x + p6
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 14 * vlen]);
        // y = y * x + p5
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 13 * vlen]);
        // y = y * x + p4
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 12 * vlen]);
        // y = y * x + p3
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 11 * vlen]);
        // y = y * x + p2
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 10 * vlen]);
        // y = y * x + p1
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 9 * vlen]);
        // y = y * x + p0 ; p0 = 0
        uni_vfmadd213ps(Vmm(8), Vmm(3), ptr[imm_addr64 + 8 * vlen]);
        //calculate ln(2) * n
        uni_vmulps(Vmm(1), Vmm(1), ptr[imm_addr64 + 3 * vlen]);
        uni_vaddps(Vmm(8), Vmm(8), Vmm(1));
        uni_vaddps(Vmm(8), Vmm(8), Vmm(5));
    }

    void soft_relu_vectorized_body() {
        uni_vmovups(Vmm(1), ptr[reg_from]);
        // compute soft relu */
        soft_relu_vectorized();
        // store result
        uni_vmovups(ptr[reg_to], Vmm(8));
    }

    void soft_relu_reminder_body() {
        movss(Xmm(1), ptr[reg_from]);
        soft_relu_vectorized();
        // store result
        movss(ptr[reg_to], Xmm(8));
    }

    void logistic_vectorized_body() {
        uni_vmovups(vmm_src, ptr[reg_from]);

        // compute exp(x)
        exp_vectorized();
        // dup exp(x)
        uni_vmovups(Vmm(14), vmm_dst);
        // (exp(x) + 1)
        uni_vaddps(Vmm(14), Vmm(14), vmm_one);
        // y = exp(x) / (exp(x) + 1)
        uni_vdivps(vmm_dst, vmm_dst, Vmm(14));

        // store result
        uni_vmovups(ptr[reg_to], vmm_dst);
    }

    void logistic_reminder_body() {
        movss(xmm_src, ptr[reg_from]);

        exp_scalar();

        movaps(Xmm(14), xmm_dst);
        // (exp(x) + 1)
        addss(Xmm(14), xmm_one);
        // y = exp(x) / (exp(x) + 1)
        divss(xmm_dst, Xmm(14));

        // store result
        movss(ptr[reg_to], xmm_dst);
    }

    void clamp_prepare_const() {
        mov(imm_addr64, float2int(desc_.alpha));
        movq(xmm_alpha, imm_addr64);
        uni_vbroadcastss(vmm_alpha, xmm_alpha);

        mov(imm_addr64, float2int(desc_.beta));
        movq(xmm_beta, imm_addr64);
        uni_vbroadcastss(vmm_beta, xmm_beta);
    }

    void clamp_vectorized_body() {
        uni_vmovups(vmm_src, ptr[reg_from]);
        // compute bounded relu */
        uni_vmaxps(vmm_src, vmm_src, vmm_beta);
        uni_vminps(vmm_src, vmm_src, vmm_alpha);
        // store result
        uni_vmovups(ptr[reg_to], vmm_src);
    }

    void clamp_reminder_body() {
        movss(xmm_src, ptr[reg_from]);
        // compute bounded relu */
        maxps(xmm_src, xmm_beta);
        minps(xmm_src, xmm_alpha);
        // store result
        movss(ptr[reg_to], xmm_src);
    }
};

} /* namespace */

template <cpu_isa_t isa>
status_t jit_uni_eltwise_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
        && utils::implication(isa > avx2, utils::one_of(desc()->alg_kind,
                eltwise_relu, eltwise_elu))
        && utils::implication(isa == sse42 || isa == avx2, utils::one_of(
                    desc()->alg_kind, eltwise_relu, eltwise_tanh, eltwise_elu,
                    eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                    eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic,
                    eltwise_clamp))
        && utils::everyone_is(data_type::f32, desc()->data_desc.data_type)
        && memory_desc_wrapper(src_pd()).is_dense()
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::jit_uni_eltwise_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr) {
    const auto &desc = *conf_.desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default:
        kernel_ = new jit_uni_kernel_fwd_f32<isa>(desc);
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_fwd_t<isa>::~jit_uni_eltwise_fwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_fwd_t<isa>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    const size_t nelems = data_d.nelems();

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = &src[start];
        arg.for_comparison = &src[start];
        arg.to = &dst[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template <cpu_isa_t isa>
status_t jit_uni_eltwise_bwd_t<isa>::pd_t::init() {
    assert(engine()->kind() == engine_kind::cpu);

    bool ok = true
        && mayiuse(isa)
        && desc()->prop_kind == prop_kind::backward_data
        && utils::one_of(desc()->alg_kind, alg_kind::eltwise_relu)
        && src_pd()->desc()->data_type == data_type::f32
        && memory_desc_wrapper(src_pd()).is_dense()
        && memory_desc_wrapper(diff_dst_pd()) == memory_desc_wrapper(src_pd())
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::jit_uni_eltwise_bwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), kernel_(nullptr) {
    const auto &desc = *conf_.desc();
    switch (desc.alg_kind) {
    case alg_kind::eltwise_relu:
        kernel_ = new jit_uni_relu_kernel_f32<isa>(desc); break;
    default: assert(!"unknown eltwise alg_kind");
    }
}

template <cpu_isa_t isa>
jit_uni_eltwise_bwd_t<isa>::~jit_uni_eltwise_bwd_t()
{ delete kernel_; }

template <cpu_isa_t isa>
void jit_uni_eltwise_bwd_t<isa>::execute_backward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper diff_data_d(conf_.diff_src_pd());

    const size_t nelems = data_d.nelems();

    src += data_d.blocking_desc().offset_padding;
    diff_dst += diff_data_d.blocking_desc().offset_padding;
    diff_src += diff_data_d.blocking_desc().offset_padding;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};

        const int cache_line = 16;

        balance211(utils::div_up(nelems, cache_line), nthr, ithr, start, end);
        start = nstl::min(nelems, start * cache_line);
        end = nstl::min(nelems, end * cache_line);

        auto arg = jit_args();
        arg.from = &diff_dst[start];
        arg.to = &diff_src[start];
        arg.for_comparison = &src[start];
        arg.work_amount = end - start;
        if (arg.work_amount)
            (*kernel_)(&arg);
    };

#   pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }
}

template struct jit_uni_eltwise_fwd_t<sse42>;
template struct jit_uni_eltwise_bwd_t<sse42>;
template struct jit_uni_eltwise_fwd_t<avx2>;
template struct jit_uni_eltwise_bwd_t<avx2>;
template struct jit_uni_eltwise_fwd_t<avx512_common>;
template struct jit_uni_eltwise_bwd_t<avx512_common>;

}
}
}
