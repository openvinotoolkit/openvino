// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOLINTBEGIN(*)

#include "jit_brgemm_src_quantization_kernel.hpp"

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

#define GET_OFF(field) offsetof(src_quantization_runtime_params_t, field)

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

template <cpu_isa_t isa>
jit_brgemm_src_quantization_kernel_t<isa>::jit_brgemm_src_quantization_kernel_t(
    const src_quantization_compile_params_t& jcp)
    : jit_src_quantization_kernel_base_t(jcp),
      jit_generator_t(jit_name()),
      vec_size(dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float)) {
    this->create_kernel();
    ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(this->jit_ker()));
}

template <cpu_isa_t isa>
void jit_brgemm_src_quantization_kernel_t<isa>::load_src(Vmm vmm_load, const Xbyak::Address& addr) {
    switch (jcp_.src_dt) {
    case data_type::f32: {
        uni_vmovups(vmm_load, addr);
        break;
    }
    default:
        assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_src_quantization_kernel_t<isa>::horiz_op(Vmm vmm_src_arg, Vmm vmm_aux_arg, op_type type) {
    auto uni_op = [&](const Xbyak::Ymm& x1, const Xbyak::Ymm& x2, const Xbyak::Operand& op) {
        if (type == op_type::max) {
            uni_vmaxps(x1, x2, op);
        } else if (type == op_type::sum) {
            uni_vpaddd(x1, x2, op);
        } else {
            assert(!"unsupported op type");
        }
    };

    if (isa == avx512_core) {
        Xbyak::Zmm zmm_src = Xbyak::Zmm(vmm_src_arg.getIdx());
        Xbyak::Zmm zmm_aux = Xbyak::Zmm(vmm_aux_arg.getIdx());
        vshuff32x4(zmm_aux, zmm_src, zmm_src, 0x4E);
        uni_op(zmm_src, zmm_src, zmm_aux);
        vshuff32x4(zmm_aux, zmm_src, zmm_src, 0xB1);
        uni_op(zmm_src, zmm_src, zmm_aux);
    } else if (isa == avx2) {
        Xbyak::Ymm ymm_src = Xbyak::Ymm(vmm_src_arg.getIdx());
        Xbyak::Ymm ymm_aux = Xbyak::Ymm(vmm_aux_arg.getIdx());
        vperm2i128(ymm_aux, ymm_src, ymm_src, 0x01);
        uni_op(ymm_src, ymm_src, ymm_aux);
    } else {
        assert(!"unsupported isa");
    }
    uni_vshufps(vmm_aux_arg, vmm_src_arg, vmm_src_arg, 0x4E);
    uni_op(vmm_src_arg, vmm_src_arg, vmm_aux_arg);
    uni_vshufps(vmm_aux_arg, vmm_src_arg, vmm_src_arg, 0xB1);
    uni_op(vmm_src_arg, vmm_src_arg, vmm_aux_arg);
}

template <cpu_isa_t isa>
void jit_brgemm_src_quantization_kernel_t<isa>::generate() {
    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src_ptr)]);
    mov(reg_qsrc, ptr[param1 + GET_OFF(qsrc_ptr)]);
    mov(reg_src_scales, ptr[param1 + GET_OFF(src_scales_ptr)]);
    mov(reg_src_grouped_sum, ptr[param1 + GET_OFF(src_grouped_sum_ptr)]);
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t src_dt_size = types::data_type_size(jcp_.src_dt);
    size_t qsrc_dt_size = types::data_type_size(jcp_.qsrc_dt);
    size_t src_scales_dt_size = types::data_type_size(data_type::f32);
    size_t src_grouped_sum_dt_size = types::data_type_size(data_type::s32);

    static const float negative_zero[16] =
        {-0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F, -0.F};

    static const float positive_one[16] =
        {1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F, 1.F};

    static const float int8_max[16] = {127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F,
                                       127.F};

    mov(reg_tmp, (size_t)negative_zero);
    uni_vmovups(vmm_sign_bit_mask(), ptr[reg_tmp]);

    mov(reg_tmp, (size_t)positive_one);
    uni_vmovups(vmm_one(), ptr[reg_tmp]);

    mov(reg_tmp, (size_t)int8_max);
    uni_vmovups(vmm_int8_max(), ptr[reg_tmp]);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, jcp_.ic_quant_block);
        jl(ic_end_label, T_NEAR);

        assert(!(jcp_.ic_quant_block % vec_size));
        assert(!(jcp_.src_sum_group_size % vec_size));

        int ic_blocks = static_cast<int>(jcp_.ic_quant_block / vec_size);
        uni_vpxor(vmm_max(), vmm_max(), vmm_max());
        for (int icb = 0; icb < ic_blocks; icb++) {
            load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
            vandnps(vmm_src(), vmm_sign_bit_mask(), vmm_src());
            uni_vmaxps(vmm_max(), vmm_max(), vmm_src());
        }

        horiz_op(vmm_max(), vmm_aux(), op_type::max);

        auto vmm_dscale = vmm_max();
        uni_vbroadcastss(vmm_dscale, Xmm(vmm_dscale.getIdx()));
        uni_vdivps(vmm_dscale, vmm_dscale, vmm_int8_max());

        uni_vdivps(vmm_qscale(), vmm_one(), vmm_dscale);

        uni_vmovss(ptr[reg_src_scales], Xmm(vmm_dscale.getIdx()));
        if (jcp_.with_src_grouped_sum) {
            uni_vxorps(vmm_src_sum_accum(), vmm_src_sum_accum(), vmm_src_sum_accum());
        }
        for (int icb = 0; icb < ic_blocks; icb++) {
            load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
            uni_vmulps(vmm_src(), vmm_src(), vmm_qscale());
            uni_vcvtps2dq(vmm_src(), vmm_src());

            if (jcp_.with_src_grouped_sum) {
                uni_vpaddd(vmm_src_sum_accum(), vmm_src_sum_accum(), vmm_src());

                if (((icb + 1) * vec_size) % jcp_.src_sum_group_size == 0) {
                    horiz_op(vmm_src_sum_accum(), vmm_aux(), op_type::sum);
                    uni_vmovss(ptr[reg_src_grouped_sum], Xmm(vmm_src_sum_accum().getIdx()));
                    uni_vxorps(vmm_src_sum_accum(), vmm_src_sum_accum(), vmm_src_sum_accum());
                    add(reg_src_grouped_sum, src_grouped_sum_dt_size);
                }
            }

            if (isa == avx512_core) {
                vpmovsdb(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], vmm_src());
            } else {
                uni_vpackssdw(vmm_src(), vmm_src(), vmm_src());
                vpermq(Ymm(vmm_src().getIdx()), Ymm(vmm_src().getIdx()), 0x08);
                uni_vpacksswb(vmm_src(), vmm_src(), vmm_src());
                vmovq(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], Xmm(vmm_src().getIdx()));
            }
        }

        sub(reg_ic_size, jcp_.ic_quant_block);
        add(reg_src, src_dt_size * jcp_.ic_quant_block);
        add(reg_qsrc, qsrc_dt_size * jcp_.ic_quant_block);
        add(reg_src_scales, src_scales_dt_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}
template struct jit_brgemm_src_quantization_kernel_t<avx512_core>;
template struct jit_brgemm_src_quantization_kernel_t<avx2>;

}  // namespace ov::intel_cpu
// NOLINTEND(*)
