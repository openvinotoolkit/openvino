// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOLINTBEGIN(*)

#include "jit_brgemm_weights_decompression_kernel.hpp"

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>

#define GET_OFF(field) offsetof(weights_decompression_runtime_params_t, field)

namespace ov::intel_cpu {

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace std::placeholders;

template <cpu_isa_t isa>
jit_brgemm_weights_decompression_kernel_t<isa>::jit_brgemm_weights_decompression_kernel_t(
    const weights_decompression_compile_params_t& jcp)
    : jit_weights_decompression_kernel_base_t(jcp),
      jit_generator_t(jit_name()),
      vec_size(dnnl::impl::cpu::x64::cpu_isa_traits_t<isa>::vlen / sizeof(float)) {
    this->create_kernel();
    ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(this->jit_ker()));
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::init_decomp_params(std::function<Vmm(int)> vmm_params,
                                                                        Xbyak::Reg64 reg_params,
                                                                        bool broadcast_values,
                                                                        data_type_t element_type) {
    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);
    for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
        if (broadcast_values) {
            switch (element_type) {
            case data_type::f32: {
                uni_vbroadcastss(vmm_params(static_cast<int>(ocb)), ptr[reg_params]);
                break;
            }
            case data_type::u8: {
                auto xmm_params = Xmm(vmm_params(static_cast<int>(ocb)).getIdx());
                auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
                movzx(reg_tmp_32, ptr[reg_params]);
                uni_vmovq(xmm_params, reg_tmp);
                uni_vcvtdq2ps(xmm_params, xmm_params);
                uni_vbroadcastss(vmm_params(static_cast<int>(ocb)), xmm_params);
                break;
            }
            case data_type::u2: {
                auto xmm_params = Xmm(vmm_params(static_cast<int>(ocb)).getIdx());
                auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
                movzx(reg_tmp_32, ptr[reg_params]);
                and_(reg_tmp_32, 0x3);
                uni_vmovq(xmm_params, reg_tmp);
                uni_vcvtdq2ps(xmm_params, xmm_params);
                uni_vbroadcastss(vmm_params(static_cast<int>(ocb)), xmm_params);
                break;
            }
            case data_type::e8m0: {
                auto xmm_params = Xmm(vmm_params(static_cast<int>(ocb)).getIdx());
                auto reg_tmp_32 = Reg32(reg_tmp.getIdx());
                movzx(reg_tmp_32, ptr[reg_params]);
                uni_vmovq(xmm_params, reg_tmp);
                uni_vpslld(xmm_params, xmm_params, 23);
                uni_vbroadcastss(vmm_params(static_cast<int>(ocb)), xmm_params);
                break;
            }
            default:
                assert(!"unsupported data type");
            }
        } else {
            const auto load_addr = ptr[reg_params + ocb * vec_size * types::data_type_size(element_type)];
            switch (element_type) {
            case data_type::f32: {
                uni_vmovups(vmm_params(static_cast<int>(ocb)), load_addr);
                break;
            }
            case data_type::u8: {
                uni_vpmovzxbd(vmm_params(static_cast<int>(ocb)), load_addr);
                uni_vcvtdq2ps(vmm_params(static_cast<int>(ocb)), vmm_params(static_cast<int>(ocb)));
                break;
            }
            case data_type::e8m0: {
                uni_vpmovzxbd(vmm_params(static_cast<int>(ocb)), load_addr);
                uni_vpslld(vmm_params(static_cast<int>(ocb)), vmm_params(static_cast<int>(ocb)), 23);
                break;
            }
            default:
                assert(!"unsupported data type");
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::load_weights(Vmm vmm_load, const Xbyak::Address& addr, int ic) {
    switch (jcp_.weights_dt) {
    case data_type::u8: {
        uni_vpmovzxbd(vmm_load, addr);
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    }
    case data_type::s8: {
        uni_vpmovsxbd(vmm_load, addr);
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    }
    case data_type::u4: {
        uni_vpmovzxbd(vmm_load, addr);
        if (ic % 2 == 0) {
            uni_vpsrld(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            uni_vpsrld(vmm_load, vmm_load, 28);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    }
    case data_type::s4: {
        uni_vpmovsxbd(vmm_load, addr);
        if (ic % 2 == 0) {
            vpsrad(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            vpsrad(vmm_load, vmm_load, 28);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    }
    case data_type::u2: {
        uni_vpmovzxbd(vmm_load, addr);
        if (ic == 0) {
            uni_vpsrld(vmm_load, vmm_load, 6);
        } else {
            uni_vpslld(vmm_load, vmm_load, 24 + 2 * ic);
            uni_vpsrld(vmm_load, vmm_load, 30);
        }
        uni_vcvtdq2ps(vmm_load, vmm_load);
        break;
    }
    case data_type::nf4: {
        uni_vpmovzxbd(vmm_load, addr);
        if (ic % 2 == 0) {
            uni_vpsrld(vmm_load, vmm_load, 4);
        } else {
            uni_vpslld(vmm_load, vmm_load, 28);
            uni_vpsrld(vmm_load, vmm_load, 28);
        }

        if (isa == avx2) {
            auto res = vmm_weights(1);
            auto mask = vmm_weights(2);
            vpcmpgtd(mask, vmm_load, vmm_mask7());
            vpermd(res, vmm_load, vmm_lookup_low());
            vpsubd(vmm_load, vmm_load, vmm_mask8());
            vpermd(vmm_load, vmm_load, vmm_lookup_high());
            vblendvps(vmm_load, res, vmm_load, mask);
        } else {
            vpermd(vmm_load, vmm_load, vmm_lookup());
        }
        break;
    }
    case data_type::f4_e2m1: {
        if (isa == avx2) {
            uni_vpmovsxbd(vmm_load, addr);
            if (ic % 2 == 0) {
                vpsrad(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                vpsrad(vmm_load, vmm_load, 28);
            }
            auto mask_reg = vmm_weights(1);
            uni_vpand(mask_reg, vmm_load, vmm_mask());
            vpermd(vmm_load, vmm_load, vmm_lookup());
            uni_vorps(vmm_load, vmm_load, mask_reg);
        } else {
            uni_vpmovzxbd(vmm_load, addr);
            if (ic % 2 == 0) {
                uni_vpsrld(vmm_load, vmm_load, 4);
            } else {
                uni_vpslld(vmm_load, vmm_load, 28);
                uni_vpsrld(vmm_load, vmm_load, 28);
            }
            vpermd(vmm_load, vmm_load, vmm_lookup());
        }
        break;
    }
    case data_type::f16: {
        vcvtph2ps(vmm_load, addr);
        break;
    }
    case data_type::bf16: {
        vpmovzxwd(vmm_load, addr);
        uni_vpslld(vmm_load, vmm_load, 16);
        break;
    }
    default:
        assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::store_weights(const Xbyak::Address& addr, Vmm vmm_store) {
    switch (jcp_.decomp_buffer_dt) {
    case data_type::f32: {
        uni_vmovups(addr, vmm_store);
        break;
    }
    case data_type::bf16: {
        Ymm ymm_store = Ymm(vmm_store.getIdx());
        vcvtneps2bf16(ymm_store, vmm_store);
        vmovdqu16(addr, ymm_store);
        break;
    }
    default:
        assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_brgemm_weights_decompression_kernel_t<isa>::generate() {
    preamble();

    mov(reg_weights, ptr[param1 + GET_OFF(weights_ptr)]);
    mov(reg_decomp_buffer, ptr[param1 + GET_OFF(decomp_buffer_ptr)]);
    if (jcp_.with_scales) {
        mov(reg_scales, ptr[param1 + GET_OFF(scales_ptr)]);
    }
    if (jcp_.with_zero_points) {
        mov(reg_zero_points, ptr[param1 + GET_OFF(zero_points_ptr)]);
    }
    mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

    if (jcp_.weights_dt == data_type::nf4) {
        static const float lookup[16] = {-1.0f,
                                         -0.6961928009986877f,
                                         -0.5250730514526367f,
                                         -0.39491748809814453f,
                                         -0.28444138169288635f,
                                         -0.18477343022823334f,
                                         -0.09105003625154495f,
                                         0.0f,
                                         0.07958029955625534f,
                                         0.16093020141124725f,
                                         0.24611230194568634f,
                                         0.33791524171829224f,
                                         0.44070982933044434f,
                                         0.5626170039176941f,
                                         0.7229568362236023f,
                                         1.0f};

        static const int32_t mask8[16] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
        static const int32_t mask7[16] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

        if (isa == avx2) {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup_low(), ptr[reg_tmp]);
            uni_vmovups(vmm_lookup_high(), ptr[reg_tmp + 8 * sizeof(float)]);
            mov(reg_tmp, (size_t)mask8);
            uni_vmovups(vmm_mask8(), ptr[reg_tmp]);
            mov(reg_tmp, (size_t)mask7);
            uni_vmovups(vmm_mask7(), ptr[reg_tmp]);
        } else {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
        }
    } else if (jcp_.weights_dt == data_type::f4_e2m1) {
        static const float lookup[16] =
            {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

        static const uint32_t mask_signed_bit[8] = {
            0x80000000,
            0x80000000,
            0x80000000,
            0x80000000,
            0x80000000,
            0x80000000,
            0x80000000,
            0x80000000,
        };

        if (isa == avx2) {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
            mov(reg_tmp, (size_t)mask_signed_bit);
            uni_vmovups(vmm_mask(), ptr[reg_tmp]);
        } else {
            mov(reg_tmp, (size_t)lookup);
            uni_vmovups(vmm_lookup(), ptr[reg_tmp]);
        }
    }

    if (jcp_.with_scales)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_scales, this, _1),
                           reg_scales,
                           jcp_.broadcast_scales,
                           jcp_.scales_dt);

    if (jcp_.with_zero_points)
        init_decomp_params(std::bind(&jit_brgemm_weights_decompression_kernel_t::vmm_zero_points, this, _1),
                           reg_zero_points,
                           jcp_.broadcast_zero_points,
                           jcp_.zero_points_dt);

    size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);

    Xbyak::Label ic_loop_label;
    Xbyak::Label ic_end_label;

    size_t weights_dt_size = types::data_type_size(jcp_.weights_dt);
    size_t typesize_scale = [&] {
        if (jcp_.weights_dt == data_type::u2) {
            return size_t(4);
        } else if (one_of(jcp_.weights_dt, data_type::nf4, data_type::s4, data_type::u4, data_type::f4_e2m1)) {
            return size_t(2);
        } else {
            return size_t(1);
        }
    }();
    size_t decomp_buf_dt_size = types::data_type_size(jcp_.decomp_buffer_dt);

    L(ic_loop_label);
    {
        cmp(reg_ic_size, 1);
        jl(ic_end_label, T_NEAR);

        if (jcp_.decomp_buffer_dt == data_type::bf16) {
            for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    size_t weights_offset = 0;
                    if (jcp_.weights_dt == data_type::u8 || jcp_.weights_dt == data_type::s8)
                        weights_offset = (ic * jcp_.oc_size + ocb * vec_size) * weights_dt_size / typesize_scale;
                    else
                        weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / typesize_scale;
                    auto vmm_load = vmm_weights(static_cast<int>(ic));
                    const auto load_addr = ptr[reg_weights + weights_offset];
                    load_weights(vmm_load, load_addr, static_cast<int>(ic));

                    if (jcp_.with_zero_points)
                        uni_vsubps(vmm_load, vmm_load, vmm_zero_points(static_cast<int>(ocb)));
                    if (jcp_.with_scales)
                        uni_vmulps(vmm_load, vmm_load, vmm_scales(static_cast<int>(ocb)));
                }

                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic += 2) {
                    auto ymm_store0 = Ymm(vmm_weights(static_cast<int>(ic)).getIdx());
                    auto ymm_store1 = Ymm(vmm_weights(static_cast<int>(ic + 1)).getIdx());
                    auto ymm_aux0 = Ymm(vmm_aux0().getIdx());
                    auto ymm_aux1 = Ymm(vmm_aux1().getIdx());

                    vcvtneps2bf16(ymm_store0, vmm_weights(static_cast<int>(ic)));
                    vcvtneps2bf16(ymm_store1, vmm_weights(static_cast<int>(ic + 1)));
                    vpunpcklwd(ymm_aux0, ymm_store0, ymm_store1);
                    vpunpckhwd(ymm_aux1, ymm_store0, ymm_store1);
                    vperm2i128(ymm_store0, ymm_aux0, ymm_aux1, 0x20);
                    vperm2i128(ymm_store1, ymm_aux0, ymm_aux1, 0x31);
                }

                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    auto ymm_store = Ymm(vmm_weights(static_cast<int>(ic)).getIdx());
                    size_t decomp_buffer_offset =
                        jcp_.weights_dt == data_type::u2
                            ? (((ic / 2) * oc_blocks_num + ocb) * 2 + (ic % 2)) * vec_size * decomp_buf_dt_size
                            : (ocb * jcp_.ic_internal_size + ic) * vec_size * decomp_buf_dt_size;
                    const auto decomp_buffer_addr = ptr[reg_decomp_buffer + decomp_buffer_offset];
                    vmovdqu16(decomp_buffer_addr, ymm_store);
                }
            }
        } else {
            for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
                for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                    size_t weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / typesize_scale;
                    const auto weights_addr = ptr[reg_weights + weights_offset];
                    load_weights(vmm_weights(0), weights_addr, static_cast<int>(ic));

                    if (jcp_.with_zero_points)
                        uni_vsubps(vmm_weights(0), vmm_weights(0), vmm_zero_points(static_cast<int>(ocb)));
                    if (jcp_.with_scales)
                        uni_vmulps(vmm_weights(0), vmm_weights(0), vmm_scales(static_cast<int>(ocb)));

                    size_t decomp_buffer_offset = (ic * jcp_.oc_size + ocb * vec_size) * decomp_buf_dt_size;
                    const auto decomp_buffer_addr = ptr[reg_decomp_buffer + decomp_buffer_offset];
                    store_weights(decomp_buffer_addr, vmm_weights(0));
                }
            }
        }

        dec(reg_ic_size);
        add(reg_weights, weights_dt_size * jcp_.oc_size * jcp_.ic_internal_size / typesize_scale);
        add(reg_decomp_buffer, decomp_buf_dt_size * jcp_.oc_size * jcp_.ic_internal_size);

        jmp(ic_loop_label, T_NEAR);
    }
    L(ic_end_label);

    postamble();
}
template struct jit_brgemm_weights_decompression_kernel_t<avx512_core>;
template struct jit_brgemm_weights_decompression_kernel_t<avx2>;

}  // namespace ov::intel_cpu
// NOLINTEND(*)
