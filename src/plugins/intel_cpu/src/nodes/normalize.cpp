// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize.h"

#include <memory>
#include <shape_inference/shape_inference_pass_through.hpp>
#include <utility>

#include "common/cpu_memcpy.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "dnnl_extension_utils.h"
#include "eltwise.h"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "fake_quantize.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "selective_build.h"
#include "utils/bfloat16.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

using namespace dnnl;

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#if defined(OPENVINO_ARCH_X86_64)
#    define GET_OFF(field) offsetof(jit_normalize_call_args, field)
#endif

namespace ov::intel_cpu::node {
namespace {

struct NormalizeKey {
    NormalizeL2::NormalizeL2Attrs attrs;
    dnnl::primitive_attr kernel_attrs;
    VectorDims dims;

    [[nodiscard]] size_t hash() const;
    bool operator==(const NormalizeKey& rhs) const;
};

size_t NormalizeKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, attrs.epsMode);
    seed = hash_combine(seed, attrs.across_spatial);
    seed = hash_combine(seed, attrs.cornerCase);
    seed = hash_combine(seed, attrs.eps);
    seed = hash_combine(seed, attrs.layout);
    seed = hash_combine(seed, attrs.input_prec.hash());
    seed = hash_combine(seed, attrs.output_prec.hash());

    seed = hash_combine(seed, get_attr_hash(*kernel_attrs.get()));
    seed = get_vector_hash(seed, dims);
    return seed;
}

bool NormalizeKey::operator==(const NormalizeKey& rhs) const {
    return (attrs.epsMode == rhs.attrs.epsMode) && (attrs.across_spatial == rhs.attrs.across_spatial) &&
           (attrs.cornerCase == rhs.attrs.cornerCase) && (attrs.eps == rhs.attrs.eps) &&
           (attrs.layout == rhs.attrs.layout) && (attrs.input_prec == rhs.attrs.input_prec) &&
           (attrs.output_prec == rhs.attrs.output_prec) && (*kernel_attrs.get() == *(rhs.kernel_attrs.get())) &&
           (dims == rhs.dims);
}

}  // namespace

#if defined(OPENVINO_ARCH_X86_64)

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

template <cpu_isa_t isa>
struct jit_uni_normalize_modulo_kernel_f32 : public jit_uni_normalize_modulo_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_modulo_kernel_f32)

    jit_uni_normalize_modulo_kernel_f32(jit_normalize_config_params jcp)
        : jit_uni_normalize_modulo_kernel(jcp),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        this->preamble();
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_modulo, ptr[reg_params + GET_OFF(modulo)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);

        Xbyak::Label modulo_loop_label;
        Xbyak::Label modulo_loop_end_label;

        uni_vpxor(vmm_sqr_sum, vmm_sqr_sum, vmm_sqr_sum);
        L(modulo_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(modulo_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            if (isa == cpu::x64::sse41 && jcp_.is_blk) {
                int sse42_offset = 4;
                load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                uni_vfmadd231ps(vmm_sqr_sum, vmm_val, vmm_val);
            }

            add(reg_src, reg_src_stride);
            sub(reg_work_amount, 1);

            jmp(modulo_loop_label, T_NEAR);
        }
        L(modulo_loop_end_label);

        if (jcp_.is_nchw && !jcp_.across_spatial) {
            uni_vmovups(ptr[reg_modulo], vmm_sqr_sum);
        } else {
            // hsum+store
            if (isa == cpu::x64::sse41) {
                hsum_store(vmm_sqr_sum);
            } else if (isa == cpu::x64::avx2) {
                auto ymm_sqr_sum = Xbyak::Ymm(vmm_sqr_sum.getIdx());
                vextractf128(xmm_aux1, ymm_sqr_sum, 0);
                vextractf128(xmm_aux2, ymm_sqr_sum, 1);
                uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            } else {
                auto zmm_sqr_sum = Xbyak::Zmm(vmm_sqr_sum.getIdx());
                vextractf32x4(xmm_aux1, zmm_sqr_sum, 0);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 1);
                uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
                vextractf32x4(xmm_aux2, zmm_sqr_sum, 2);
                vextractf32x4(xmm_aux3, zmm_sqr_sum, 3);
                uni_vaddps(xmm_aux2, xmm_aux2, xmm_aux3);
                uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
                hsum_store(xmm_aux1);
            }
        }

        this->postamble();
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_src_stride = r10;
    Xbyak::Reg64 reg_modulo = rbp;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_val = Vmm(0);
    Vmm vmm_sqr_sum = Vmm(1);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(2);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(3);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(4);

    inline void hsum_store(Xbyak::Xmm xmm_sqr_sum) {
        uni_vmovshdup(xmm_aux3, xmm_sqr_sum);            //  sqrt_sum:1,2,3,4; aux3:2,2,4,4
        uni_vaddps(xmm_sqr_sum, xmm_sqr_sum, xmm_aux3);  //  sqrt_sum:1+2,2+2,3+4,4+4
        uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_sqr_sum);   //  aux3:3+4,4+4,4,4
        uni_vaddps(xmm_sqr_sum, xmm_sqr_sum, xmm_aux3);  //  sqrt_sum:1+2+3+4,...
        uni_vmovss(ptr[reg_modulo], xmm_sqr_sum);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address& op, memory::data_type src_dt) {
        switch (src_dt) {
        case memory::data_type::f32:
        case memory::data_type::s32:
            uni_vmovups(vmm_src, op);
            break;
        case memory::data_type::bf16:
            uni_vpmovzxwd(vmm_src, op);
            uni_vpslld(vmm_src, vmm_src, 16);
            break;
        case memory::data_type::s8:
            uni_vpmovsxbd(vmm_src, op);
            break;
        case memory::data_type::u8:
            uni_vpmovzxbd(vmm_src, op);
            break;
        default:
            assert(!"unknown dst_dt");
        }
        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(vmm_src, vmm_src);
        }
    }
};

// dst = src * modulo_inv
template <cpu_isa_t isa>
struct jit_uni_normalize_kernel_f32 : public jit_uni_normalize_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_normalize_kernel_f32)

    explicit jit_uni_normalize_kernel_f32(jit_normalize_config_params jcp, const dnnl_primitive_attr& attr)
        : jit_uni_normalize_kernel(jcp, attr),
          jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto& p = attr_.post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector<isa>>(this,
                                                                                            post_op.eltwise.alg,
                                                                                            post_op.eltwise.alpha,
                                                                                            post_op.eltwise.beta,
                                                                                            post_op.eltwise.scale,
                                                                                            data_type::f32));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(this, post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(this,
                                                                                                          post_op,
                                                                                                          vmm_d_weights,
                                                                                                          vmm_d_bias,
                                                                                                          reg_d_weights,
                                                                                                          reg_d_bias));
            }
        }

        if (mayiuse(avx512_core)) {
            uni_vcvtneps2bf16 = std::make_unique<jit_uni_vcvtneps2bf16>(this, isa);
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_fused_factor, ptr[reg_params + GET_OFF(fused_factor)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        if (attr_.post_ops_.len() != 0) {
            mov(reg_post_ops_data, ptr[reg_params + GET_OFF(post_op_data)]);
            mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
        }
        if (isa == avx512_core) {
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        }

        if (jcp_.is_nchw) {
            normalize_nchw();
        } else if (jcp_.is_blk) {
            normalize_blk();
        } else if (jcp_.is_nhwc) {
            normalize_nhwc();
        }

        this->postamble();

        if (uni_vcvtneps2bf16) {
            uni_vcvtneps2bf16->emit_data();
        }
        for (auto& inj : eltwise_injectors) {
            inj->prepare_table();
        }
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_fused_factor = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Reg8 reg_tmp_8 = r14b;
    Reg32 reg_tmp_32 = r14d;
    Reg64 reg_tmp_64 = r14;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_post_ops_data = rbx;
    Xbyak::Reg64 reg_d_weights = reg_tmp_64;
    Xbyak::Reg64 reg_d_bias = rdx;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_scale = Vmm(1);
    Xmm xmm_scale = Xmm(1);
    Vmm vmm_modulo = Vmm(2);
    Xmm xmm_modulo = Xmm(2);
    Vmm vmm_fused_factor = Vmm(3);
    Xmm xmm_fused_factor = Xmm(3);
    Vmm vmm_fused_factor2 = Vmm(4);
    Xmm xmm_fused_factor2 = Xmm(4);

    Vmm vmm_d_weights = Vmm(5);
    Vmm vmm_d_bias = Vmm(6);
    Vmm vmm_zero = Vmm(7);

    std::unique_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16 = nullptr;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    inline void normalize_nchw() {
        if (jcp_.across_spatial) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);  // for channel_shared: false or true.
        }

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = jcp_.src_dt == memory::data_type::bf16 ? 16 : (vlen / sizeof(float));
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
            } else {
                uni_vmovups(vmm_fused_factor, ptr[reg_fused_factor]);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                add(reg_fused_factor, vlen);
            }
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 1);
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            if (jcp_.across_spatial) {
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
            } else {
                load_scalar(xmm_fused_factor, ptr[reg_fused_factor], memory::data_type::f32);
                uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);
                add(reg_fused_factor, step * sizeof(float));
            }
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 1);  // vector and boradcast
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    inline void normalize_nhwc() {
        uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);

        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        int step = jcp_.src_dt == memory::data_type::bf16 ? 16 : (vlen / sizeof(float));
        L(main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, vlen);  // out channel offset of fused ops weights in byte
            }
            store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(main_loop_label, T_NEAR);
        }
        L(main_loop_end_label);

        step = 1;
        L(tail_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(tail_loop_end_label, T_NEAR);

            load_scalar(xmm_val, ptr[reg_src], jcp_.src_dt);
            uni_vmulps(xmm_val, xmm_val, xmm_fused_factor);

            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(jcp_.dst_dt, 0);
                add(reg_oc_off, step * sizeof(float));
            }
            store_scalar(ptr[reg_dst], xmm_val, jcp_.dst_dt);

            add(reg_src, step * jcp_.src_data_size);
            add(reg_dst, step * jcp_.dst_data_size);
            sub(reg_work_amount, step);

            jmp(tail_loop_label, T_NEAR);
        }
        L(tail_loop_end_label);
    }

    // tails with padding as a vector for normalize.
    inline void normalize_blk() {
        size_t blk_size = 0;
        size_t simd_w = 0;
        if (isa == cpu::x64::avx512_core) {
            blk_size = simd_w = 16;
        } else if (isa == cpu::x64::avx2) {
            blk_size = simd_w = 8;
        } else {
            blk_size = 8;
            simd_w = 4;
        }
        bool is_sse42 = (isa == cpu::x64::sse41);

        if (jcp_.across_spatial) {
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);

                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    if (attr_.post_ops_.len() != 0) {
                        add(reg_oc_off, sse42_offset * sizeof(float));
                        apply_post_ops(jcp_.dst_dt, 0);
                        sub(reg_oc_off, sse42_offset * sizeof(float));
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, blk_size * jcp_.src_data_size);
                add(reg_dst, blk_size * jcp_.dst_data_size);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
        } else {  // across_saptail is flase
            uni_vbroadcastss(vmm_fused_factor, ptr[reg_fused_factor]);
            size_t src_stride = jcp_.w * jcp_.h * blk_size * jcp_.src_data_size;
            size_t dst_stride = jcp_.w * jcp_.h * blk_size * jcp_.dst_data_size;

            Xbyak::Label norm_loop_label;
            Xbyak::Label norm_loop_end_label;

            L(norm_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(norm_loop_end_label, T_NEAR);

                load_vector(vmm_val, ptr[reg_src], jcp_.src_dt);
                uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(jcp_.dst_dt, 0);
                    add(reg_oc_off, vlen);  // vlen is related isa
                }
                store_vector(ptr[reg_dst], vmm_val, jcp_.dst_dt);

                if (is_sse42) {
                    int sse42_offset = 4;
                    load_vector(vmm_val, ptr[reg_src + sse42_offset * jcp_.src_data_size], jcp_.src_dt);
                    uni_vmulps(vmm_val, vmm_val, vmm_fused_factor);  // bc once
                    if (attr_.post_ops_.len() != 0) {
                        apply_post_ops(jcp_.dst_dt, 0);
                        add(reg_oc_off, vlen);  // vlen is related isa
                    }
                    store_vector(ptr[reg_dst + sse42_offset * jcp_.dst_data_size], vmm_val, jcp_.dst_dt);
                }
                add(reg_src, src_stride);
                add(reg_dst, dst_stride);

                sub(reg_work_amount, 1);
                jmp(norm_loop_label, T_NEAR);
            }
            L(norm_loop_end_label);
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address& op, memory::data_type src_dt) {
        switch (src_dt) {
        case memory::data_type::f32:
        case memory::data_type::s32:
            uni_vmovups(vmm_src, op);
            break;
        case memory::data_type::bf16:
            uni_vpmovzxwd(vmm_src, op);
            uni_vpslld(vmm_src, vmm_src, 16);
            break;
        case memory::data_type::s8:
            uni_vpmovsxbd(vmm_src, op);
            break;
        case memory::data_type::u8:
            uni_vpmovzxbd(vmm_src, op);
            break;
        default:
            assert(!"unknown dst_dt");
        }
        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(vmm_src, vmm_src);
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address& op, memory::data_type src_dt) {
        switch (src_dt) {
        case memory::data_type::f32:
        case memory::data_type::s32:
            uni_vmovss(xmm_src, op);
            break;
        case memory::data_type::bf16:
            uni_vpinsrw(xmm_src, xmm_src, op, 0x0);
            uni_vpslld(xmm_src, xmm_src, 16);
            break;
        case memory::data_type::s8:
            movsx(reg_tmp_32, op);
            uni_vmovq(xmm_src, reg_tmp_64);
            break;
        case memory::data_type::u8:
            movzx(reg_tmp_32, op);
            uni_vmovq(xmm_src, reg_tmp_64);
            break;
        default:
            assert(!"unknown dst_dt");
        }

        if (!isFloatCompatible(src_dt)) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address& op, Vmm vmm_dst, memory::data_type dst_dt) {
        auto ymm_dst = Ymm(vmm_dst.getIdx());
        auto xmm_dst = Xmm(vmm_dst.getIdx());

        if (dst_dt == memory::data_type::f32) {
            uni_vmovups(op, vmm_dst);
        } else if (dst_dt == memory::data_type::bf16) {
            uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                         {static_cast<size_t>(ymm_dst.getIdx())});
            vmovdqu16(op, ymm_dst);
        } else if (dst_dt == memory::data_type::u8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::x64::avx512_core) {
                vpmaxsd(vmm_dst, vmm_dst, vmm_zero);
                vpmovusdb(op, vmm_dst);
            } else {
                uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41) {
                    vpermq(ymm_dst, ymm_dst, 0x08);
                }
                uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41) {
                    vmovq(op, xmm_dst);
                } else {
                    movd(op, xmm_dst);
                }
            }
        } else if (dst_dt == memory::data_type::s8) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
            if (isa == cpu::x64::avx512_core) {
                vpmovsdb(op, vmm_dst);
            } else {
                uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41) {
                    vpermq(ymm_dst, ymm_dst, 0x08);
                }
                uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                if (isa != cpu::x64::sse41) {
                    vmovq(op, xmm_dst);
                } else {
                    movd(op, xmm_dst);
                }
            }
        }
    }

    inline void store_scalar(const Xbyak::Address& op, Xmm xmm_dst, memory::data_type dst_dt) {
        if (!isFloatCompatible(dst_dt)) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
        case memory::data_type::f32:
        case memory::data_type::s32:
            uni_vmovss(op, xmm_dst);
            break;
        case memory::data_type::bf16:
            uni_vpsrld(xmm_dst, xmm_dst, 16);
            uni_vpextrw(op, xmm_dst, 0x0);
            break;
        case memory::data_type::s8:
            uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
            movq(reg_tmp_64, xmm_dst);
            mov(op, reg_tmp_8);
            break;
        case memory::data_type::u8:
            uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
            movq(reg_tmp_64, xmm_dst);
            mov(op, reg_tmp_8);
            break;
        default:
            assert(!"unknown dst_dt");
        }
    }

    // scalar: load scalar to xmm, process on xmm with padded param, store xmm to scalar.
    // is_broadcast for broadcasting param for depth_wise and quantize, for fusion with plain layout.
    void apply_post_ops(memory::data_type dst_dt, bool is_broadcast) {
        const auto& p = attr_.post_ops_;
        size_t eltwise_inj_idx = 0;
        size_t depthwise_inj_idx = 0;
        size_t quantization_inj_idx = 0;
        int post_ops_data_offset = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                if (eltwise_injectors.size() <= eltwise_inj_idx || eltwise_injectors[eltwise_inj_idx] == nullptr) {
                    assert(!"Invalid eltwise injectors.");
                }
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                if (depthwise_injectors.size() <= depthwise_inj_idx ||
                    depthwise_injectors[depthwise_inj_idx] == nullptr) {
                    assert(!"Invalid depthwise injectors.");
                }
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);

                // weight and bias is padding. scalar as vector.
                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_val.getIdx(),
                                                                             vmm_val.getIdx() + 1,
                                                                             reg_d_weights,
                                                                             reg_d_weights,
                                                                             is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                if (quantization_injectors.size() <= quantization_inj_idx ||
                    quantization_injectors[quantization_inj_idx] == nullptr) {
                    assert(!"Invalid quantization injectors.");
                }
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_dt) || i != p.len() - 1;

                int s_idx = vmm_val.getIdx();

                const Xbyak::RegExp quant_arg_base = reg_post_ops_data + post_ops_data_offset;
                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(quant_arg_base, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(s_idx, s_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(quant_arg_base, reg_oc_off);
                quantization_injectors[quantization_inj_idx]
                    ->compute_input_scale_shift(s_idx, s_idx + 1, 0, do_rounding, 0, is_broadcast);

                if (do_dequantization) {
                    quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(quant_arg_base,
                                                                                               reg_oc_off);
                    quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(s_idx,
                                                                                             s_idx + 1,
                                                                                             0,
                                                                                             0,
                                                                                             is_broadcast);
                }

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }
};
#endif
bool NormalizeL2::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto norm = ov::as_type_ptr<const ov::op::v0::NormalizeL2>(op);
        if (!norm) {
            errorMessage = "Only opset1 NormalizeL2 operation is supported";
            return false;
        }

        const auto inputRank = norm->get_input_partial_shape(DATA).size();
        if (inputRank < 2 || inputRank > 4) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(inputRank);
            return false;
        }

        auto axesNode = ov::as_type_ptr<const ov::op::v0::Constant>(norm->get_input_node_shared_ptr(AXES));
        if (!axesNode) {
            errorMessage = "Supports only constant 'axes' input";
            return false;
        }

        if (axesNode->get_type_info() != ov::op::v0::Constant::get_type_info_static()) {
            // TODO [DS]: Add 'axes' input dynamism support
            errorMessage = "Doesn't support dynamic 'axes' input";
            return false;
        }

        const auto isSupportedAxes = [](const std::vector<size_t>& axes, const size_t inputRank) {
            if (axes.size() == 1 && axes[0] == 1) {
                return true;
            }
            if (axes.size() == inputRank - 1) {
                auto sortAxes = axes;
                std::sort(sortAxes.begin(), sortAxes.end());
                for (size_t i = 0; i < sortAxes.size(); i++) {
                    if (sortAxes[i] != i + 1) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        };

        const auto axes = axesNode->cast_vector<size_t>();
        if (!isSupportedAxes(axes, inputRank) && ov::shape_size(axesNode->get_shape()) != 0) {
            errorMessage = "Doesn't support reduction axes: " + vec2str(axes);
            return false;
        }

        const auto mode = norm->get_eps_mode();
        if (!one_of(mode, ov::op::EpsMode::ADD, ov::op::EpsMode::MAX)) {
            errorMessage = "Doesn't support eps_mode: " + ov::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NormalizeL2::NormalizeL2(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (inputShapes.size() != 2 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges");
    }

    if (getInputShapeAtPort(DATA).getRank() > 4 || getInputShapeAtPort(DATA).getRank() < 2) {
        THROW_CPU_NODE_ERR("has invalid input shape. Normalize supports from 2D to 4D blobs.");
    }

    auto norm = ov::as_type_ptr<const ov::op::v0::NormalizeL2>(op);
    attrs.eps = norm->get_eps();
    attrs.epsMode = norm->get_eps_mode() == ov::op::EpsMode::MAX ? NormEpsMode::MAX : NormEpsMode::ADD;
    attrs.across_spatial = ov::shape_size(op->get_input_shape(AXES)) != 1;
    // One of the corner cases is when axes is an empty list,
    // then we divide each input element by itself resulting value 1 for all non-zero elements
    attrs.cornerCase = ov::shape_size(op->get_input_shape(AXES)) == 0;
}

void NormalizeL2::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type inputPrecision = getOriginalInputPrecisionAtPort(DATA);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(DATA);

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    if (inputPrecision == ov::element::bf16 || outputPrecision == ov::element::bf16) {
        if (!mayiuse(avx512_core)) {
            inputPrecision = outputPrecision = ov::element::f32;
        } else {
            inputPrecision = outputPrecision = ov::element::bf16;
        }
    }

    if (one_of(ov::element::f16, inputPrecision, outputPrecision) && mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = ov::element::f32;
    }

    if (!one_of(inputPrecision,
                ov::element::f32,
                ov::element::bf16,
                ov::element::f16,
                ov::element::i8,
                ov::element::u8)) {
        THROW_CPU_NODE_ERR("has unsupported input precision: ", inputPrecision);
    }
    if (!one_of(outputPrecision,
                ov::element::f32,
                ov::element::bf16,
                ov::element::f16,
                ov::element::i8,
                ov::element::u8)) {
        THROW_CPU_NODE_ERR("has unsupported output precision: ", outputPrecision);
    }

    attrs.input_prec = inputPrecision;
    attrs.output_prec = outputPrecision;
    attrs.src_data_size = inputPrecision.size();
    attrs.dst_data_size = outputPrecision.size();

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && attrs.src_data_size == attrs.dst_data_size &&
                        getParentEdgeAt(DATA)->getParent()->getChildEdges().size() == 1;

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, impl_desc_type impl_type) {
        auto a = creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(DATA));
        config.inConfs[0].setMemDesc(std::move(a));
        a = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::i32, getInputShapeAtPort(AXES));
        config.inConfs[1].setMemDesc(std::move(a));
        a = creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(DATA));
        config.outConfs[0].setMemDesc(std::move(a));
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };

    impl_desc_type impl_type = impl_desc_type::unknown;

    // only plain layout support when w/o sse42
    if (getInputShapeAtPort(DATA).getRank() == 4 && !attrs.cornerCase) {
        if (mayiuse(cpu::x64::sse41)) {
            pushDesc(LayoutType::nspc, impl_type);
            if (mayiuse(cpu::x64::avx512_core)) {
                pushDesc(LayoutType::nCsp16c, impl_type);
            } else {
                pushDesc(LayoutType::nCsp8c, impl_type);
            }
        }
    }
    if (canBeInplace) {
        config.inConfs[0].inPlace(0);
    }
    pushDesc(LayoutType::ncsp, impl_type);
}

bool NormalizeL2::canFuse(const NodePtr& node) const {
    return !attrs.cornerCase && canFuseSimpleOperation(node);
}

void NormalizeL2::setPostOps(dnnl::primitive_attr& kernel_attrs, const VectorDims& dims, bool initWeights) {
    dnnl::post_ops ops;

    postOpsDataPtrs.clear();
    for (auto& node : fusedWith) {
        int channelAxis = 1;

        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs, channelAxis);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, dims, postOpsDataPtrs, channelAxis);
            continue;
        }

        THROW_CPU_NODE_ERR("Fusing of ",
                           NameFromType(node->getType()),
                           " operation to ",
                           NameFromType(this->getType()),
                           " node is not implemented");
    }

    kernel_attrs.set_post_ops(ops);
}

void NormalizeL2::createPrimitive() {
    auto dstMemPtr = getDstMemoryAtPort(DATA);
    auto srcMemPtr = getSrcMemoryAtPort(DATA);
    if (!dstMemPtr) {
        THROW_CPU_NODE_ERR("can't get destination memory");
    }
    if (!srcMemPtr) {
        THROW_CPU_NODE_ERR("can't get input memory");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has nullable preferable primitive descriptor");
    }

    if (!attrs.cornerCase) {
        if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
            attrs.layout = LayoutType::ncsp;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c)) {
            attrs.layout = LayoutType::nCsp8c;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c)) {
            attrs.layout = LayoutType::nCsp16c;
        } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
            attrs.layout = LayoutType::nspc;
        } else {
            THROW_CPU_NODE_ERR("has selected layout which is not supported");
        }
    }

    if (inputShapesDefined()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

bool NormalizeL2::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool NormalizeL2::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

void NormalizeL2::prepareParams() {
    const auto& dims = getSrcMemoryAtPort(DATA)->getStaticDims();

    setPostOps(kernel_attrs, dims, true);

    NormalizeKey key = {attrs, kernel_attrs, dims};

    auto builder = [](const NormalizeKey& key) -> std::shared_ptr<NormalizeL2::NormalizeL2Executor> {
        return NormalizeL2Executor::getNormalizeL2Executor(key.attrs, key.kernel_attrs, key.dims);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);

    if (!result.first) {
        THROW_CPU_NODE_ERR("Primitive descriptor was not found.");
    }

    execPtr = result.first;
}

void NormalizeL2::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void NormalizeL2::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("doesn't have a compiled executor.");
    }

    const auto* src_ptr = getSrcDataAtPortAs<const uint8_t>(DATA);
    auto* dst_ptr = getDstDataAtPortAs<uint8_t>(DATA);
    execPtr->exec(src_ptr, dst_ptr, postOpsDataPtrs.data());
}

// *====================* CornerCase *===================*

template <typename in_data_t, typename out_data_t>
class NormalizeL2::NormalizeL2CornerCaseExecutor : public NormalizeL2::NormalizeL2Executor {
public:
    NormalizeL2CornerCaseExecutor(const VectorDims& dims)
        : workAmount(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>())) {}

    void exec(const uint8_t* src_ptr, uint8_t* dst_ptr, const void** post_ops_data) override {
        normalize(reinterpret_cast<const in_data_t*>(src_ptr), reinterpret_cast<out_data_t*>(dst_ptr));
    }

private:
    void normalize(const in_data_t* src_data, out_data_t* dst_data) {
        parallel_for(workAmount, [&](size_t i) {
            dst_data[i] = src_data[i] == 0 ? 0 : 1;
        });
    }

    size_t workAmount = 0lu;
};

// *=================* *======* *=================*

// *=================* JIT case *=================*
#if defined(OPENVINO_ARCH_X86_64)
template <typename in_data_t, typename out_data_t>
class NormalizeL2::NormalizeL2JitExecutor : public NormalizeL2::NormalizeL2Executor {
public:
    NormalizeL2JitExecutor(const NormalizeL2Attrs& attrs_,
                           const dnnl::primitive_attr& kernel_attrs,
                           const VectorDims& dims)
        : attrs(attrs_) {
        if (attrs.layout != LayoutType::ncsp && attrs.layout != LayoutType::nspc &&
            attrs.layout != LayoutType::nCsp8c && attrs.layout != LayoutType::nCsp16c) {
            OPENVINO_THROW("Normalaize2L executor has selected layout which is not supported");
        }

        jcp.src_dt = DnnlExtensionUtils::ElementTypeToDataType(attrs.input_prec);
        jcp.dst_dt = DnnlExtensionUtils::ElementTypeToDataType(attrs.output_prec);
        jcp.src_data_size = attrs.input_prec.size();
        jcp.dst_data_size = attrs.output_prec.size();
        jcp.across_spatial = attrs.across_spatial;

        jcp.is_nchw = (attrs.layout == LayoutType::ncsp);
        jcp.is_nhwc = (attrs.layout == LayoutType::nspc);
        jcp.is_blk = (attrs.layout == LayoutType::nCsp8c || attrs.layout == LayoutType::nCsp16c);

        size_t dims_size = dims.size();
        jcp.n = dims[0];
        jcp.c = dims[1];
        jcp.h = (dims_size > 2) ? dims[2] : 1lu;
        jcp.w = (dims_size > 3) ? dims[3] : 1lu;

        if (mayiuse(cpu::x64::avx512_core)) {
            blk_size = 16;
            normalize_modulo_kernel = std::make_shared<jit_uni_normalize_modulo_kernel_f32<cpu::x64::avx512_core>>(jcp);
            normalize_kernel =
                std::make_shared<jit_uni_normalize_kernel_f32<cpu::x64::avx512_core>>(jcp, *kernel_attrs.get());
        } else if (mayiuse(cpu::x64::avx2)) {
            blk_size = 8;
            normalize_modulo_kernel = std::make_shared<jit_uni_normalize_modulo_kernel_f32<cpu::x64::avx2>>(jcp);
            normalize_kernel = std::make_shared<jit_uni_normalize_kernel_f32<cpu::x64::avx2>>(jcp, *kernel_attrs.get());
        } else if (mayiuse(cpu::x64::sse41)) {
            blk_size = jcp.is_blk ? 8 : 4;
            normalize_modulo_kernel = std::make_shared<jit_uni_normalize_modulo_kernel_f32<cpu::x64::sse41>>(jcp);
            normalize_kernel =
                std::make_shared<jit_uni_normalize_kernel_f32<cpu::x64::sse41>>(jcp, *kernel_attrs.get());
        } else {
            OPENVINO_THROW("Jit Executor for NormalizeL2 cannot create kernels!");
        }

        if (normalize_kernel) {
            normalize_kernel->create_ker();
        }

        if (normalize_modulo_kernel) {
            normalize_modulo_kernel->create_ker();
        }
    }

    void exec(const uint8_t* src_ptr, uint8_t* dst_ptr, const void** post_ops_data) override {
        if (jcp.is_nchw) {
            normalize_nchw(reinterpret_cast<const in_data_t*>(src_ptr),
                           reinterpret_cast<out_data_t*>(dst_ptr),
                           post_ops_data);
        } else if (jcp.is_nhwc) {
            normalize_nhwc(reinterpret_cast<const in_data_t*>(src_ptr),
                           reinterpret_cast<out_data_t*>(dst_ptr),
                           post_ops_data);
        } else if (jcp.is_blk) {
            normalize_blk(reinterpret_cast<const in_data_t*>(src_ptr),
                          reinterpret_cast<out_data_t*>(dst_ptr),
                          post_ops_data);
        }
    }

private:
    void normalize_nchw(const in_data_t* src_data, out_data_t* dst_data, const void** post_ops_data) {
        const size_t spatial_dims = jcp.h * jcp.w;
        for (size_t b = 0lu; b < jcp.n; b++) {
            const in_data_t* src_data_b = src_data + b * jcp.c * spatial_dims;
            out_data_t* dst_data_b = dst_data + b * jcp.c * spatial_dims;
            if (attrs.across_spatial) {
                // modulo
                float addition_identity = 0.0f;
                float modulo = 0.0f;
                modulo = parallel_sum(jcp.c, addition_identity, [&](int ic) -> float {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    float modulo_kernel = 0.0f;
                    float modulo_tail = 0.0f;
                    size_t tail_start = 0;

                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bc;
                    arg.modulo = static_cast<float*>(&modulo_kernel);
                    arg.src_stride = blk_size * sizeof(in_data_t);
                    arg.work_amount = (spatial_dims) / blk_size;
                    (*normalize_modulo_kernel)(&arg);

                    tail_start = (spatial_dims / blk_size) * blk_size;

                    // tail
                    for (size_t tail = tail_start; tail < spatial_dims; tail++) {
                        modulo_tail += src_data_bc[tail] * src_data_bc[tail];
                    }
                    return modulo_kernel + modulo_tail;
                });

                float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                // normalize
                parallel_for(jcp.c, [&](size_t ic) {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    out_data_t* dst_data_bc = dst_data_b + ic * spatial_dims;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bc;
                    arg.dst = dst_data_bc;
                    arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast once
                    arg.oc_off = ic * sizeof(float);
                    arg.work_amount = static_cast<size_t>(spatial_dims);
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            } else {  // across_spatial: false
                // moduloM
                std::vector<float> moduloM(spatial_dims, 0.f);
                size_t blocks_num = div_up(spatial_dims, blk_size);
                parallel_for(blocks_num, [&](size_t ib) {
                    const in_data_t* src_data_b_ib = src_data_b + ib * blk_size;
                    size_t min_cb = (std::min)(blk_size, spatial_dims - (ib * blk_size));
                    if (min_cb == blk_size) {
                        auto arg = jit_normalize_call_args();
                        arg.src = src_data_b_ib;
                        arg.modulo = static_cast<float*>(&moduloM[ib * blk_size]);
                        arg.src_stride = spatial_dims * sizeof(in_data_t);
                        arg.work_amount = jcp.c;
                        (*normalize_modulo_kernel)(&arg);
                    } else {
                        for (size_t c = 0; c < jcp.c; c++) {
                            const in_data_t* src_data_b_ib_c = src_data_b_ib + spatial_dims * c;
                            for (size_t blk = 0; blk < min_cb; blk++) {
                                moduloM[ib * blk_size + blk] += src_data_b_ib_c[blk] * src_data_b_ib_c[blk];
                            }
                        }
                    }
                });

                for (size_t m = 0; m < spatial_dims; m++) {
                    moduloM[m] = 1.0f / (std::sqrt(epsApply(moduloM[m], attrs.epsMode, attrs.eps)));
                }

                // normalize
                parallel_for(jcp.c, [&](size_t ic) {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    out_data_t* dst_data_bc = dst_data_b + ic * spatial_dims;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bc;
                    arg.dst = dst_data_bc;
                    arg.fused_factor = static_cast<float*>(&moduloM[0]);  // ld dynamic
                    arg.oc_off = ic * sizeof(float);
                    arg.work_amount = static_cast<size_t>(spatial_dims);
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            }
        }
    }

    void normalize_nhwc(const in_data_t* src_data, out_data_t* dst_data, const void** post_ops_data) {
        const size_t spatial_dims = jcp.h * jcp.w;
        const size_t c_w_dims = jcp.c * jcp.w;
        for (size_t b = 0lu; b < jcp.n; b++) {
            const in_data_t* src_data_b = src_data + b * jcp.c * spatial_dims;
            out_data_t* dst_data_b = dst_data + b * jcp.c * spatial_dims;
            if (attrs.across_spatial) {
                // modulo
                float addition_identity = 0;
                float modulo = 0.0f;
                modulo = parallel_sum(jcp.h, addition_identity, [&](int ih) -> float {
                    size_t tail_start = 0;
                    const in_data_t* src_data_bh = src_data_b + ih * c_w_dims;
                    float modulo_kernel = 0.f;
                    float modulo_tail = 0.f;

                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bh;
                    arg.modulo = static_cast<float*>(&modulo_kernel);
                    arg.src_stride = blk_size * sizeof(in_data_t);
                    arg.work_amount = c_w_dims / blk_size;
                    (*normalize_modulo_kernel)(&arg);

                    tail_start = (c_w_dims / blk_size) * blk_size;

                    // tail
                    for (size_t tail = tail_start; tail < c_w_dims; tail++) {
                        modulo_tail += src_data_bh[tail] * src_data_bh[tail];
                    }
                    return modulo_kernel + modulo_tail;
                });

                float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                // normalize
                parallel_for2d(jcp.h, jcp.w, [&](int ih, int iw) {
                    const in_data_t* src_data_bhw = src_data_b + ih * c_w_dims + iw * jcp.c;
                    out_data_t* dst_data_bhw = dst_data_b + ih * c_w_dims + iw * jcp.c;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bhw;
                    arg.dst = dst_data_bhw;
                    arg.fused_factor = static_cast<float*>(&modulo_inv);  // bc static
                    arg.oc_off = 0;
                    arg.work_amount = static_cast<size_t>(jcp.c);
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            } else {  // for across_spatial=false
                parallel_for2d(jcp.h, jcp.w, [&](int ih, int iw) {
                    // modulo
                    float modulo = 0.f;
                    const in_data_t* src_data_bhw = src_data_b + ih * c_w_dims + iw * jcp.c;
                    out_data_t* dst_data_bhw = dst_data_b + ih * c_w_dims + iw * jcp.c;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bhw;
                    arg.modulo = static_cast<float*>(&modulo);
                    arg.src_stride = blk_size * sizeof(in_data_t);
                    arg.work_amount = jcp.c / blk_size;
                    (*normalize_modulo_kernel)(&arg);

                    size_t tail_start = (jcp.c / blk_size) * blk_size;

                    // for tail
                    for (size_t c = tail_start; c < jcp.c; c++) {
                        modulo += src_data_bhw[c] * src_data_bhw[c];
                    }

                    float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                    // normalize
                    arg.dst = dst_data_bhw;
                    arg.fused_factor = static_cast<float*>(&modulo_inv);  // bc static
                    arg.work_amount = jcp.c;
                    arg.oc_off = 0;
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            }
        }
    }

    void normalize_blk(const in_data_t* src_data, out_data_t* dst_data, const void** post_ops_data) {
        const size_t CB = div_up(jcp.c, blk_size);
        const size_t spatial_dims = jcp.h * jcp.w;
        const size_t w_blk_dims = jcp.w * blk_size;
        for (size_t b = 0lu; b < jcp.n; b++) {
            const in_data_t* src_data_b = src_data + b * CB * spatial_dims * blk_size;
            out_data_t* dst_data_b = dst_data + b * CB * spatial_dims * blk_size;
            if (attrs.across_spatial) {
                // modulo
                float modulo = 0.0f;
                float addition_identity = 0.0f;
                modulo = parallel_sum2d(CB, jcp.h, addition_identity, [&](size_t cb, size_t h) -> float {
                    // handle W * blk_size data
                    const in_data_t* src_data_b_cb_h = src_data_b + cb * spatial_dims * blk_size + h * w_blk_dims;
                    size_t min_cb = (std::min)(blk_size, jcp.c - cb * blk_size);
                    float modulo_w_blk = 0.0f;
                    if (min_cb == blk_size) {
                        auto arg = jit_normalize_call_args();
                        arg.src = src_data_b_cb_h;
                        arg.modulo = static_cast<float*>(&modulo_w_blk);
                        arg.src_stride = blk_size * sizeof(in_data_t);
                        arg.work_amount = jcp.w;
                        (*normalize_modulo_kernel)(&arg);
                    } else {
                        for (size_t w = 0; w < jcp.w; w++) {
                            const in_data_t* src_data_b_cb_h_w = src_data_b_cb_h + w * blk_size;
                            for (size_t c = 0; c < min_cb; c++) {
                                modulo_w_blk += src_data_b_cb_h_w[c] * src_data_b_cb_h_w[c];
                            }
                        }
                    }
                    return modulo_w_blk;
                });

                float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                // normalize
                parallel_for2d(CB, jcp.h, [&](size_t cb, size_t h) {
                    const in_data_t* src_data_b_cb_h = src_data_b + cb * spatial_dims * blk_size + h * w_blk_dims;
                    out_data_t* dst_data_b_cb_h = dst_data_b + cb * spatial_dims * blk_size + h * w_blk_dims;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_b_cb_h;
                    arg.dst = dst_data_b_cb_h;
                    arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast once
                    arg.work_amount = static_cast<size_t>(jcp.w);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            } else {  // across_spatial: false
                parallel_for2d(jcp.h, jcp.w, [&](size_t ih, size_t iw) {
                    // modulo
                    float modulo = 0.0f;
                    const in_data_t* src_data_bhw = src_data_b + ih * w_blk_dims + iw * blk_size;
                    out_data_t* dst_data_bhw = dst_data_b + ih * w_blk_dims + iw * blk_size;
                    auto arg = jit_normalize_call_args();
                    arg.src = src_data_bhw;
                    arg.modulo = static_cast<float*>(&modulo);
                    arg.src_stride = blk_size * spatial_dims * sizeof(in_data_t);
                    arg.work_amount = jcp.c / blk_size;  // CB or CB-1
                    (*normalize_modulo_kernel)(&arg);
                    // for tail
                    size_t padding = CB * blk_size - jcp.c;
                    if (padding > 0) {
                        size_t tail = blk_size - padding;
                        const in_data_t* src_data_bhw_lastCB = src_data_bhw + (CB - 1) * blk_size * spatial_dims;
                        for (size_t c = 0; c < tail; c++) {
                            modulo += src_data_bhw_lastCB[c] * src_data_bhw_lastCB[c];
                        }
                    }

                    float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                    // normalize
                    arg.dst = dst_data_bhw;
                    arg.fused_factor = static_cast<float*>(&modulo_inv);  // broadcast
                    arg.work_amount = CB;
                    arg.oc_off = 0;
                    arg.post_op_data = post_ops_data;
                    (*normalize_kernel)(&arg);
                });
            }
        }
    }

    size_t blk_size = 1lu;
    jit_normalize_config_params jcp = {};
    NormalizeL2Attrs attrs;

    std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel;
    std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel;
};
#endif
// *=================* *======* *=================*

// *=============* Reference case *===============*

template <typename in_data_t, typename out_data_t>
class NormalizeL2::NormalizeL2ReferenceExecutor : public NormalizeL2::NormalizeL2Executor {
public:
    NormalizeL2ReferenceExecutor(const NormalizeL2Attrs& attrs,
                                 const dnnl::primitive_attr& kernel_attrs,
                                 VectorDims dims)
        : dims(std::move(dims)),
          kernel_attrs(kernel_attrs),
          attrs(attrs) {
        if (attrs.layout != LayoutType::ncsp) {
            OPENVINO_THROW("Reference Executor of 'NormalizeL2' supports only ncsp layout!");
        }

        const auto& p = (*kernel_attrs.get()).post_ops_;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors_ref.push_back(std::make_shared<cpu::ref_eltwise_scalar_fwd_t>(post_op.eltwise.alg,
                                                                                                post_op.eltwise.alpha,
                                                                                                post_op.eltwise.beta,
                                                                                                post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors_ref.push_back(
                    std::make_shared<cpu::ref_depthwise_scalar_fwd_t>(post_op.depthwise.alg));
            }
        }
    }

    void exec(const uint8_t* src_ptr, uint8_t* dst_ptr, const void** post_ops_data) override {
        normalize_nchw_ref(reinterpret_cast<const in_data_t*>(src_ptr),
                           reinterpret_cast<out_data_t*>(dst_ptr),
                           post_ops_data);
    }

private:
    void normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data, const void** post_ops_data) {
        size_t dims_size = dims.size();
        const size_t N = dims[0];
        const size_t C = dims[1];
        const size_t H = (dims_size > 2) ? dims[2] : 1lu;
        const size_t W = (dims_size > 3) ? dims[3] : 1lu;
        const size_t spatial_dims = H * W;
        for (size_t b = 0lu; b < N; b++) {
            const in_data_t* src_data_b = src_data + b * C * spatial_dims;
            out_data_t* dst_data_b = dst_data + b * C * spatial_dims;
            if (attrs.across_spatial) {
                // modulo
                float addition_identity = 0.0f;
                float modulo = 0.0f;
                modulo = parallel_sum(C, addition_identity, [&](int ic) -> float {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    float modulo_c = 0.0f;
                    for (size_t m = 0; m < spatial_dims; m++) {
                        modulo_c += src_data_bc[m] * src_data_bc[m];
                    }
                    return modulo_c;
                });

                float modulo_inv = 1.0f / (std::sqrt(epsApply(modulo, attrs.epsMode, attrs.eps)));

                // normalize
                parallel_for(C, [&](size_t ic) {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    out_data_t* dst_data_bc = dst_data_b + ic * spatial_dims;
                    for (size_t m = 0; m < spatial_dims; m++) {
                        float dst_value = src_data_bc[m] * modulo_inv;
                        apply_post_ops_scalar(dst_value, ic, post_ops_data);
                        if (attrs.output_prec == ov::element::u8) {
                            dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                        } else {
                            dst_data_bc[m] = dst_value;
                        }
                    }
                });
            } else {  // across_spatial: false
                // moduloM
                std::vector<float> moduloM(spatial_dims, 0.f);
                parallel_for(H, [&](size_t ih) {
                    size_t offset_h = ih * W;
                    const in_data_t* src_data_b_ih = src_data_b + offset_h;
                    for (size_t c = 0; c < C; c++) {
                        const in_data_t* src_data_b_ih_c = src_data_b_ih + spatial_dims * c;
                        for (size_t w = 0; w < W; w++) {
                            moduloM[offset_h + w] += src_data_b_ih_c[w] * src_data_b_ih_c[w];
                        }
                    }
                });

                for (size_t m = 0; m < spatial_dims; m++) {
                    moduloM[m] = 1.0f / (std::sqrt(epsApply(moduloM[m], attrs.epsMode, attrs.eps)));
                }

                // normalize
                parallel_for(C, [&](size_t ic) {
                    const in_data_t* src_data_bc = src_data_b + ic * spatial_dims;
                    out_data_t* dst_data_bc = dst_data_b + ic * spatial_dims;
                    for (size_t m = 0; m < spatial_dims; m++) {
                        float dst_value = src_data_bc[m] * moduloM[m];
                        apply_post_ops_scalar(dst_value, ic, post_ops_data);
                        if (attrs.output_prec == ov::element::u8) {
                            dst_data_bc[m] = (dst_value >= 0) ? dst_value : 0;
                        } else {
                            dst_data_bc[m] = dst_value;
                        }
                    }
                });
            }
        }
    }

    inline void apply_post_ops_scalar(float& dst_value, int index_c, const void** post_ops_data_) {
        const auto& p = (*kernel_attrs.get()).post_ops_;
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        // reinterpret cast from (pointer to const void) to (pointer to const pointer to const float)
        const auto** post_ops_data = reinterpret_cast<const float**>(post_ops_data_);
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.entry_[i];
            if (post_op.is_eltwise()) {
                dst_value = eltwise_injectors_ref[eltwise_inj_idx]->compute_scalar(dst_value);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                auto depthwise_base = *post_ops_data;
                auto depthwise_weights = depthwise_base + post_op.depthwise.offset[post_op.depthwise.scales] + index_c;
                auto depthwise_bias = depthwise_base + post_op.depthwise.offset[post_op.depthwise.shifts] + index_c;

                dst_value = depthwise_injectors_ref[depthwise_inj_idx]->compute_scalar(dst_value,
                                                                                       depthwise_weights,
                                                                                       depthwise_bias);

                depthwise_inj_idx++;
                post_ops_data++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || attrs.output_prec == ov::element::f32 || i != p.len() - 1;

                auto quant = post_op.quantization;

                using quantization_fields = post_ops_t::entry_t::quantization_t::quantization_fields;
                auto dataVal = [&](const quantization_fields& field) -> float {
                    auto dataPtr = *post_ops_data + post_op.quantization.offset[field];
                    const int channelIdx = quant.per_channel[field] ? index_c : 0;
                    return dataPtr[channelIdx];
                };

                float crop_low = dataVal(quant.crop_low);
                float crop_high = dataVal(quant.crop_high);
                float input_scale = dataVal(quant.inp_scale);
                float input_shift = dataVal(quant.inp_shift);

                dst_value = nstl::min(crop_high, nstl::max(crop_low, dst_value));
                dst_value = dst_value * input_scale + input_shift;

                if (do_rounding) {
                    dst_value = roundf(dst_value);
                }

                if (do_dequantization) {
                    float output_scale = dataVal(quant.output_scale);
                    float output_shift = dataVal(quant.output_shift);
                    dst_value = dst_value * output_scale + output_shift;
                }

                post_ops_data++;
            }
        }
    }

    VectorDims dims;
    dnnl::primitive_attr kernel_attrs;
    NormalizeL2Attrs attrs;

    std::vector<std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref;
    std::vector<std::shared_ptr<dnnl::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref;
};

// *=================* *======* *=================*

std::shared_ptr<NormalizeL2::NormalizeL2Executor> NormalizeL2::NormalizeL2Executor::getNormalizeL2Executor(
    const NormalizeL2Attrs& attrs,
    const dnnl::primitive_attr& kernel_attrs,
    const VectorDims& dims) {
    NormalizeContext ctx = {nullptr, attrs, kernel_attrs, dims};

    OV_SWITCH(intel_cpu,
              NormalizeExecutorCreation,
              ctx,
              std::tie(attrs.input_prec, attrs.output_prec),
              OV_CASE2(ov::element::u8, ov::element::u8, uint8_t, uint8_t),
              OV_CASE2(ov::element::i8, ov::element::u8, int8_t, uint8_t),
              OV_CASE2(ov::element::f32, ov::element::u8, float, uint8_t),
              OV_CASE2(ov::element::u8, ov::element::i8, uint8_t, int8_t),
              OV_CASE2(ov::element::i8, ov::element::i8, int8_t, int8_t),
              OV_CASE2(ov::element::f32, ov::element::i8, float, int8_t),
              OV_CASE2(ov::element::u8, ov::element::f32, uint8_t, float),
              OV_CASE2(ov::element::i8, ov::element::f32, int8_t, float),
              OV_CASE2(ov::element::f32, ov::element::f32, float, float),
              OV_CASE2(ov::element::bf16, ov::element::bf16, bfloat16_t, bfloat16_t),
              OV_CASE2(ov::element::f16, ov::element::f16, dnnl::impl::float16_t, dnnl::impl::float16_t));

    return ctx.executor;
}

template <typename in_data_t, typename out_data_t>
std::shared_ptr<NormalizeL2::NormalizeL2Executor> NormalizeL2::NormalizeL2Executor::makeExecutor(
    const NormalizeL2Attrs& attrs,
    const dnnl::primitive_attr& kernel_attrs,
    const VectorDims& dims) {
    if (attrs.cornerCase) {
        return std::make_shared<NormalizeL2CornerCaseExecutor<in_data_t, out_data_t>>(dims);
#if defined(OPENVINO_ARCH_X86_64)
    }
    if (mayiuse(cpu::x64::sse41)) {
        return std::make_shared<NormalizeL2JitExecutor<in_data_t, out_data_t>>(attrs, kernel_attrs, dims);
#endif
    } else if (attrs.layout == LayoutType::ncsp) {
        return std::make_shared<NormalizeL2ReferenceExecutor<in_data_t, out_data_t>>(attrs, kernel_attrs, dims);
    } else {
        OPENVINO_THROW("'NormalizeL2' cannot create Executor");
    }
}

bool NormalizeL2::created() const {
    return getType() == Type::NormalizeL2;
}

}  // namespace ov::intel_cpu::node
