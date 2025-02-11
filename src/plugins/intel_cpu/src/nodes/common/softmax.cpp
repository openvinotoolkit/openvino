// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_bf16_emitters.hpp"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

#define GET_OFF(field) offsetof(jit_args_softmax, field)

namespace ov::intel_cpu {

struct jit_args_softmax {
    const void* src;
    void* dst;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
};

struct jit_softmax_config_params {
    ov::element::Type src_dt;
    ov::element::Type dst_dt;
};

struct jit_uni_softmax_kernel {
    void (*ker_)(const jit_args_softmax*){nullptr};

    void operator()(const jit_args_softmax* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_softmax_kernel() = default;
    virtual ~jit_uni_softmax_kernel() = default;

    virtual void create_ker() = 0;
};
#if defined(OPENVINO_ARCH_X86_64)
template <cpu_isa_t isa>
struct jit_uni_softmax_kernel_f32 : public jit_uni_softmax_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_softmax_kernel_f32)

    jit_uni_softmax_kernel_f32(jit_softmax_config_params jcp)
        : jit_uni_softmax_kernel(),
          jit_generator(jit_name()),
          jcp_(jcp) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        exp_injector.reset(
            new jit_uni_eltwise_injector<isa>(this, dnnl::impl::alg_kind::eltwise_exp, 0.f, 0.f, 1.0f, data_type::f32));

        if (mayiuse(avx512_core)) {
            uni_vcvtneps2bf16 = std::make_unique<jit_uni_vcvtneps2bf16>(this, isa);
        }

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_dst_stride, ptr[reg_params + GET_OFF(dst_stride)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        Xbyak::Label max_loop_label;
        Xbyak::Label max_loop_end_label;
        Xbyak::Label exp_loop_label;
        Xbyak::Label exp_loop_end_label;
        Xbyak::Label div_loop_label;
        Xbyak::Label div_loop_end_label;

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        load_vector(vmm_max, ptr[aux_reg_src], jcp_.src_dt);
        L(max_loop_label);
        {
            cmp(aux_reg_work_amount, 0);
            jle(max_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[aux_reg_src], jcp_.src_dt);

            if (isa == x64::sse41) {
                uni_vmovups(vmm_mask, vmm_val);
                uni_vcmpgtps(vmm_mask, vmm_mask, vmm_max);
            } else if (isa == x64::avx2) {
                uni_vcmpgtps(vmm_mask, vmm_val, vmm_max);
            } else {
                vcmpps(k_mask, vmm_val, vmm_max, _cmp_nle_us);
            }

            if (isa == x64::avx512_core) {
                vptestmd(k_mask, vmm_mask, vmm_mask);
                vblendmps(vmm_max | k_mask, vmm_max, vmm_val);
            } else {
                uni_vblendvps(vmm_max, vmm_max, vmm_val, vmm_mask);
            }

            add(aux_reg_src, reg_src_stride);
            sub(aux_reg_work_amount, 1);

            jmp(max_loop_label, T_NEAR);
        }

        L(max_loop_end_label);

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_src, reg_src);
        mov(aux_reg_dst, reg_dst);
        uni_vpxor(vmm_exp_sum, vmm_exp_sum, vmm_exp_sum);
        L(exp_loop_label);
        {
            cmp(aux_reg_work_amount, 0);
            jle(exp_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[aux_reg_src], jcp_.src_dt);

            uni_vsubps(vmm_val, vmm_val, vmm_max);
            exp_injector->compute_vector_range(vmm_val.getIdx(), vmm_val.getIdx() + 1);
            uni_vaddps(vmm_exp_sum, vmm_exp_sum, vmm_val);

            store_vector(ptr[aux_reg_dst], vmm_val, jcp_.dst_dt);

            add(aux_reg_src, reg_src_stride);
            add(aux_reg_dst, reg_dst_stride);
            sub(aux_reg_work_amount, 1);

            jmp(exp_loop_label, T_NEAR);
        }

        L(exp_loop_end_label);

        mov(aux_reg_work_amount, reg_work_amount);
        mov(aux_reg_dst, reg_dst);
        L(div_loop_label);
        {
            cmp(aux_reg_work_amount, 0);
            jle(div_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[aux_reg_dst], jcp_.dst_dt);

            uni_vdivps(vmm_val, vmm_val, vmm_exp_sum);

            store_vector(ptr[aux_reg_dst], vmm_val, jcp_.dst_dt);

            add(aux_reg_dst, reg_dst_stride);
            sub(aux_reg_work_amount, 1);

            jmp(div_loop_label, T_NEAR);
        }

        L(div_loop_end_label);

        this->postamble();

        if (uni_vcvtneps2bf16) {
            uni_vcvtneps2bf16->emit_data();
        }

        exp_injector->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 aux_reg_src = r13;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 aux_reg_dst = r15;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 aux_reg_work_amount = r12;
    Xbyak::Reg64 reg_src_stride = r14;
    Xbyak::Reg64 reg_dst_stride = r10;
    Xbyak::Reg64 reg_params = abi_param1;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_val = Vmm(1);
    Vmm vmm_max = Vmm(2);
    Vmm vmm_exp_sum = Vmm(3);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    std::unique_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;

    std::shared_ptr<jit_uni_eltwise_injector<isa>> exp_injector;

    jit_softmax_config_params jcp_;

    inline void load_vector(Vmm vmm_src, const Xbyak::Address& op, ov::element::Type src_dt) {
        switch (src_dt) {
        case ov::element::f32:
            uni_vmovups(vmm_src, op);
            break;
        case ov::element::bf16:
            vpmovzxwd(vmm_src, op);
            uni_vpslld(vmm_src, vmm_src, 16);
            break;
        default:
            assert(!"unknown src_dt");
        }
    }
    inline void store_vector(const Xbyak::Address& op, Vmm vmm_dst, ov::element::Type dst_dt) {
        auto ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());

        switch (dst_dt) {
        case ov::element::f32:
            uni_vmovups(op, vmm_dst);
            break;
        case ov::element::bf16:
            uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                         {static_cast<size_t>(ymm_dst.getIdx())});
            vmovdqu16(op, ymm_dst);
            break;
        default:
            assert(!"unknown dst_dt");
        }
    }
};
#endif
SoftmaxGeneric::SoftmaxGeneric(ov::element::Type inpPrc, ov::element::Type outPrc)
    : input_prec(inpPrc),
      output_prec(outPrc) {
    if (ov::element::bf16 == output_prec) {
        if (!mayiuse(avx512_core)) {
            OPENVINO_THROW("SoftmaxGeneric doesn't support BF16 precision on this target.");
        }
    }

    block_size = 1;
#if defined(OPENVINO_ARCH_X86_64)
    auto jcp = jit_softmax_config_params();
    jcp.src_dt = inpPrc;
    jcp.dst_dt = outPrc;

    if (mayiuse(x64::avx512_core)) {
        softmax_kernel = std::make_shared<jit_uni_softmax_kernel_f32<x64::avx512_core>>(jcp);
        block_size = 16;
    } else if (mayiuse(x64::avx2)) {
        softmax_kernel = std::make_shared<jit_uni_softmax_kernel_f32<x64::avx2>>(jcp);
        block_size = 8;
    } else if (mayiuse(x64::sse41)) {
        softmax_kernel = std::make_shared<jit_uni_softmax_kernel_f32<x64::sse41>>(jcp);
        block_size = 4;
    }
    if (softmax_kernel) {
        softmax_kernel->create_ker();
    }
#endif
}

template <typename in_data_t, typename out_data_t>
void SoftmaxGeneric::calculate(const in_data_t* src_data, out_data_t* dst_data, int B, int C, int H, int W) {
    for (int b = 0; b < B; b++) {
        int tail_start = 0;

        if (softmax_kernel) {
            int blocks_num = H * W / block_size;

            parallel_for(blocks_num, [&](int ib) {
                auto arg = jit_args_softmax();

                arg.src = src_data + b * C * H * W + ib * block_size;
                arg.dst = dst_data + b * C * H * W + ib * block_size;
                arg.src_stride = static_cast<size_t>(static_cast<size_t>(H) * W * sizeof(in_data_t));
                arg.dst_stride = static_cast<size_t>(static_cast<size_t>(H) * W * sizeof(out_data_t));
                arg.work_amount = static_cast<size_t>(C);

                (*softmax_kernel)(&arg);
            });

            tail_start = (H * W / block_size) * block_size;
        }

        parallel_for(H * W - tail_start, [&](int i) {
            int offset = i + tail_start;
            float max = src_data[b * C * H * W + offset];
            for (int c = 0; c < C; c++) {
                float val = src_data[b * C * H * W + c * H * W + offset];
                if (val > max) {
                    max = val;
                }
            }

            float expSum = 0;
            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + offset] = exp(src_data[b * C * H * W + c * H * W + offset] - max);
                expSum += dst_data[b * C * H * W + c * H * W + offset];
            }

            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + offset] = dst_data[b * C * H * W + c * H * W + offset] / expSum;
            }
        });
    }
}

void SoftmaxGeneric::execute(const uint8_t* src_data, uint8_t* dst_data, int B, int C, int H, int W) {
    if (ov::element::f32 == input_prec) {
        auto float_src_data = reinterpret_cast<const float*>(src_data);
        if (ov::element::f32 == output_prec) {
            auto float_dst_data = reinterpret_cast<float*>(dst_data);
            calculate(float_src_data, float_dst_data, B, C, H, W);
        } else if (ov::element::bf16 == output_prec) {
            auto bf16_dst_data = reinterpret_cast<bfloat16_t*>(dst_data);
            calculate(float_src_data, bf16_dst_data, B, C, H, W);
        } else {
            OPENVINO_THROW("Unsupported output precision: ", output_prec.get_type_name());
        }
    } else if (ov::element::bf16 == input_prec) {
        auto bf16_src_data = reinterpret_cast<const bfloat16_t*>(src_data);
        if (ov::element::f32 == output_prec) {
            auto float_dst_data = reinterpret_cast<float*>(dst_data);
            calculate(bf16_src_data, float_dst_data, B, C, H, W);
        } else if (ov::element::bf16 == output_prec) {
            auto bf16_dst_data = reinterpret_cast<bfloat16_t*>(dst_data);
            calculate(bf16_dst_data, bf16_dst_data, B, C, H, W);
        } else {
            OPENVINO_THROW("Unsupported output precision: ", output_prec.get_type_name());
        }
    } else {
        OPENVINO_THROW("Unsupported input precision: ", input_prec.get_type_name());
    }
}

}  // namespace ov::intel_cpu
