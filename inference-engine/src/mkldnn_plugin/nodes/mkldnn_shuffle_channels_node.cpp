// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_shuffle_channels_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <set>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"
#include "common/cpu_memcpy.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_shuffle_channels_call_args, field)
#define GET_SHUFFLE_INDEX(idx) (idx * group_size % shuffle_size + idx * group_size / shuffle_size)
#define GET_PTR(in_work_offset, out_work_offset, work_amount) const uint8_t *in_p  = in_ptr  +  (in_work_offset) * (work_amount) * data_size; \
                                                                    uint8_t *out_p = out_ptr + (out_work_offset) * (work_amount) * data_size

#define vmm_idx(i) Vmm(i)
#define vmm_src(i) Vmm(jcp_.group + i)
#define vmm_dst(i) Vmm(2 * jcp_.group + i)

const int tab_avx2_group_2[] = { // 00 01 00 00 02 03 00 00
                                 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000002, 0x00000003, 0x00000000, 0x00000000,
                                 // 04 05 00 00 06 07 00 00
                                 0x00000004, 0x00000005, 0x00000000, 0x00000000, 0x00000006, 0x00000007, 0x00000000, 0x00000000 };

const int tab_avx2_group_4[] = { // 00 00 00 00 01 00 00 00
                                 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                 // 02 00 00 00 03 00 00 00
                                 0x00000002, 0x00000000, 0x00000000, 0x00000000, 0x00000003, 0x00000000, 0x00000000, 0x00000000,
                                 // 04 00 00 00 05 00 00 00
                                 0x00000004, 0x00000000, 0x00000000, 0x00000000, 0x00000005, 0x00000000, 0x00000000, 0x00000000,
                                 // 06 00 00 00 07 00 00 00
                                 0x00000006, 0x00000000, 0x00000000, 0x00000000, 0x00000007, 0x00000000, 0x00000000, 0x00000000 };

const int tab_avx512_group_2[] = { // 00 16 01 17 02 18 03 19 04 20 05 21 06 22 07 23
                                   0x00000000, 0x00000010, 0x00000001, 0x00000011, 0x00000002, 0x00000012, 0x00000003, 0x00000013,
                                   0x00000004, 0x00000014, 0x00000005, 0x00000015, 0x00000006, 0x00000016, 0x00000007, 0x00000017,
                                   // 08 24 09 25 10 26 11 27 12 28 13 29 14 30 15 31
                                   0x00000008, 0x00000018, 0x00000009, 0x00000019, 0x0000000a, 0x0000001a, 0x0000000b, 0x0000001b,
                                   0x0000000c, 0x0000001c, 0x0000000d, 0x0000001d, 0x0000000e, 0x0000001e, 0x0000000f, 0x0000001f };

const int tab_avx512_group_4[] = { // 00 16 00 00 01 17 00 00 02 18 00 00 03 19 00 00
                                   0x00000000, 0x00000010, 0x00000000, 0x00000000, 0x00000001, 0x00000011, 0x00000000, 0x00000000,
                                   0x00000002, 0x00000012, 0x00000000, 0x00000000, 0x00000003, 0x00000013, 0x00000000, 0x00000000,
                                   // 04 20 00 00 05 21 00 00 06 22 00 00 07 23 00 00
                                   0x00000004, 0x00000014, 0x00000000, 0x00000000, 0x00000005, 0x00000015, 0x00000000, 0x00000000,
                                   0x00000006, 0x00000016, 0x00000000, 0x00000000, 0x00000007, 0x00000017, 0x00000000, 0x00000000,
                                   // 08 24 00 00 09 25 00 00 10 26 00 00 11 27 00 00
                                   0x00000008, 0x00000018, 0x00000000, 0x00000000, 0x00000009, 0x00000019, 0x00000000, 0x00000000,
                                   0x0000000a, 0x0000001a, 0x00000000, 0x00000000, 0x0000000b, 0x0000001b, 0x00000000, 0x00000000,
                                   // 12 28 00 00 13 29 00 00 14 30 00 00 15 31 00 00
                                   0x0000000c, 0x0000001c, 0x00000000, 0x00000000, 0x0000000d, 0x0000001d, 0x00000000, 0x00000000,
                                   0x0000000e, 0x0000001e, 0x00000000, 0x00000000, 0x0000000f, 0x0000001f, 0x00000000, 0x00000000 };

template <cpu_isa_t isa>
struct jit_uni_shuffle_channels_kernel_f32 : public jit_uni_shuffle_channels_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_channels_kernel_f32)

    explicit jit_uni_shuffle_channels_kernel_f32(jit_shuffle_channels_config_params jcp)
    : jit_uni_shuffle_channels_kernel(jcp), jit_generator() {
        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16.reset(new jit_emu_vcvtneps2bf16(this, isa, nullptr));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_index, ptr[reg_params + GET_OFF(index)]);
        mov(reg_tab_idx, ptr[reg_params + GET_OFF(tab_idx)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        if (jcp_.shuffle_innermost) {
            if (jcp_.permute_mode)
                shuffle_loop_permute();
            else
                shuffle_loop_gather();
        } else {
            shuffle_loop();
        }

        this->postamble();

        if (!mayiuse(avx512_core_bf16) && mayiuse(avx512_core))
            emu_vcvtneps2bf16->emit_table();

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_index = r10;
    Xbyak::Reg64 reg_tab_idx = r13;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg8 reg_tmp_8 = r12b;
    Xbyak::Reg32 reg_tmp_32 = r12d;
    Xbyak::Reg64 reg_tmp_64 = r12;

    Vmm vmm_val = Vmm(0);
    Xmm xmm_val = Xmm(0);
    Vmm vmm_index = Vmm(1);
    Xmm xmm_index = Xmm(1);
    Vmm vmm_mask = Vmm(2);
    Xmm xmm_mask = Xmm(2);
    Vmm vmm_zero = Vmm(3);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    std::unique_ptr<jit_emu_vcvtneps2bf16> emu_vcvtneps2bf16;

    inline void shuffle_loop() {
        Xbyak::Label shuffle_main_loop_label;
        Xbyak::Label shuffle_main_loop_end_label;
        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
        L(shuffle_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(shuffle_main_loop_end_label, T_NEAR);

            load_vector(vmm_val, ptr[reg_src], jcp_.data_type);
            store_vector(ptr[reg_dst], vmm_val, jcp_.data_type);
            if (isa == cpu::sse42) {
                load_vector(vmm_val, ptr[reg_src + 4 * jcp_.data_size], jcp_.data_type);
                store_vector(ptr[reg_dst + 4 * jcp_.data_size], vmm_val, jcp_.data_type);
            }

            add(reg_src, step * jcp_.data_size);
            add(reg_dst, step * jcp_.data_size);
            sub(reg_work_amount, step);

            jmp(shuffle_main_loop_label, T_NEAR);
        }
        L(shuffle_main_loop_end_label);

        if (jcp_.layout != ShuffleChannelsLayoutType::blocked_layout) {
            Xbyak::Label shuffle_tail_loop_label;
            Xbyak::Label shuffle_tail_loop_end_label;
            step = 1;
            L(shuffle_tail_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(shuffle_tail_loop_end_label, T_NEAR);

                load_scalar(xmm_val, ptr[reg_src], jcp_.data_type);
                store_scalar(ptr[reg_dst], xmm_val, jcp_.data_type);

                add(reg_src, step * jcp_.data_size);
                add(reg_dst, step * jcp_.data_size);
                sub(reg_work_amount, step);

                jmp(shuffle_tail_loop_label, T_NEAR);
            }
            L(shuffle_tail_loop_end_label);
        }
    }

    // essential condition: group equals 2 or 4
    inline void shuffle_loop_permute() {
        Xbyak::Label shuffle_main_loop_label;
        Xbyak::Label shuffle_main_loop_end_label;
        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);

        for (size_t i = 0; i < jcp_.group; i++)
            load_vector(vmm_idx(i), ptr[reg_tab_idx + i * step * sizeof(int)], memory::s32);

        size_t times = 1;
        int src_stride = step * jcp_.data_size;
        int dst_stride = jcp_.group * step * jcp_.data_size;
        int work_amount_stride = jcp_.group * step;
        if (jcp_.layout == ShuffleChannelsLayoutType::blocked_layout) {
            times = jcp_.channel_batch / jcp_.group;
            dst_stride = step * jcp_.data_size;
            work_amount_stride = jcp_.channel_batch * step;
        }
        int src_off[jcp_.group], dst_off[jcp_.group];
        for (size_t i = 0; i < jcp_.group; i++) {
            src_off[i] = (jcp_.shuffle_size * i / jcp_.group) * jcp_.data_size;
            dst_off[i] = i * step * jcp_.data_size;
        }

        L(shuffle_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(shuffle_main_loop_end_label, T_NEAR);

            for (size_t k = 0; k < times; k++) {
                if (jcp_.layout == ShuffleChannelsLayoutType::blocked_layout) {
                    for (size_t i = 0; i < jcp_.group; i++) {
                        src_off[i] = (i * times + k) * jcp_.shuffle_stride * jcp_.data_size;
                        dst_off[i] = (jcp_.group * k + i) * jcp_.shuffle_stride * jcp_.data_size;
                    }
                }
                for (size_t i = 0; i < jcp_.group; i++)
                    load_vector(vmm_src(i), ptr[reg_src + src_off[i]], jcp_.data_type);
                for (size_t i = 0; i < jcp_.group; i++) {
                    // avx512
                    if (isa == cpu::avx512_common) {
                        uni_vmovups(vmm_dst(0), vmm_idx(i));
                        vpermi2ps(vmm_dst(0), vmm_src(0), vmm_src(1));

                        if (jcp_.group == 4) {
                            uni_vmovups(vmm_dst(1), vmm_idx(i));
                            vpermi2ps(vmm_dst(1), vmm_src(2), vmm_src(2));
                            vshufps(vmm_dst(0), vmm_dst(0), vmm_dst(1), 0x44);
                        }
                    } else if (isa == cpu::avx2) { // avx2
                        vpermps(Ymm(vmm_dst(0).getIdx()), Ymm(vmm_idx(i).getIdx()), vmm_src(0));
                        vpermps(Ymm(vmm_dst(1).getIdx()), Ymm(vmm_idx(i).getIdx()), vmm_src(1));
                        vunpcklps(vmm_dst(0), vmm_dst(0), vmm_dst(1));

                        if (jcp_.group == 4) {
                            vpermps(Ymm(vmm_dst(2).getIdx()), Ymm(vmm_idx(i).getIdx()), vmm_src(2));
                            vpermps(Ymm(vmm_dst(3).getIdx()), Ymm(vmm_idx(i).getIdx()), vmm_src(3));
                            vunpcklps(vmm_dst(2), vmm_dst(2), vmm_dst(3));
                            vshufps(vmm_dst(0), vmm_dst(0), vmm_dst(2), 0x44);
                        }
                    }

                    store_vector(ptr[reg_dst + dst_off[i]], vmm_dst(0), jcp_.data_type);
                }
            }

            add(reg_src, src_stride);
            add(reg_dst, dst_stride);
            sub(reg_work_amount, work_amount_stride);

            jmp(shuffle_main_loop_label, T_NEAR);
        }
        L(shuffle_main_loop_end_label);
    }

    inline void shuffle_loop_gather() {
        Xbyak::Label shuffle_main_loop_label;
        Xbyak::Label shuffle_main_loop_end_label;
        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
        L(shuffle_main_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(shuffle_main_loop_end_label, T_NEAR);

            load_gathered_vector(vmm_val, ptr[reg_index], jcp_.data_type);
            store_vector(ptr[reg_dst], vmm_val, jcp_.data_type);
            if (isa == cpu::sse42) {
                load_gathered_vector(vmm_val, ptr[reg_index + 4 * sizeof(int)], jcp_.data_type);
                store_vector(ptr[reg_dst + 4 * jcp_.data_size], vmm_val, jcp_.data_type);
            }

            add(reg_dst, step * jcp_.data_size);
            add(reg_index, step * sizeof(int));
            sub(reg_work_amount, step);

            jmp(shuffle_main_loop_label, T_NEAR);
        }
        L(shuffle_main_loop_end_label);

        if (jcp_.layout != ShuffleChannelsLayoutType::blocked_layout) {
            Xbyak::Label shuffle_tail_loop_label;
            Xbyak::Label shuffle_tail_loop_end_label;
            step = 1;
            L(shuffle_tail_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(shuffle_tail_loop_end_label, T_NEAR);

                load_gathered_scalar(xmm_val, ptr[reg_index], jcp_.data_type);
                store_scalar(ptr[reg_dst], xmm_val, jcp_.data_type);

                add(reg_dst, step * jcp_.data_size);
                add(reg_index, step * sizeof(int));
                sub(reg_work_amount, step);

                jmp(shuffle_tail_loop_label, T_NEAR);
            }
            L(shuffle_tail_loop_end_label);
        }
    }

    inline void load_gathered_vector(Vmm vmm_val, const Xbyak::Address &op, memory::data_type src_dt) {
        uni_vmovdqu(vmm_index, op);
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                if (isa == cpu::avx512_common) {
                    vcmpps(k_mask, vmm_mask, vmm_mask, _cmp_eq_oq);
                    vgatherdps(vmm_val | k_mask, ptr[reg_src + vmm_index]);
                } else if (isa == cpu::avx2) {
                    uni_vpcmpeqd(vmm_mask, vmm_mask, vmm_mask);
                    vgatherdps(vmm_val, ptr[reg_src + vmm_index], vmm_mask);
                } else if (isa == cpu::sse42) {
                    pack_gathered_vector(vmm_val, vmm_index, src_dt);
                }
                break;
            case memory::bf16:
            case memory::s8:
            case memory::u8:
                pack_gathered_vector(vmm_val, vmm_index, src_dt);
                break;
            default:
                assert(!"unknown src_dt");
        }
    }

    inline void pack_gathered_vector(Vmm vmm_val, Vmm vmm_index, memory::data_type src_dt) {
        sub(rsp, vlen);
        uni_vmovdqu(ptr[rsp], vmm_index);
        int repeats = vlen / sizeof(float);
        for (size_t i = 0; i < repeats; i++) {
            mov(reg_tmp_64.cvt32(), ptr[rsp + i * sizeof(int)]);
            Xbyak::Address table_idx = ptr[reg_src + reg_tmp_64];
            switch (src_dt) {
                case memory::f32:
                case memory::s32:
                    mov(reg_tmp_64.cvt32(), table_idx);
                    mov(ptr[rsp + i * sizeof(int)], reg_tmp_64.cvt32());
                    break;
                case memory::bf16:
                    mov(reg_tmp_64.cvt16(), table_idx);
                    mov(ptr[rsp + i * sizeof(bfloat16_t)], reg_tmp_64.cvt16());
                    break;
                case memory::s8:
                case memory::u8:
                    mov(reg_tmp_64.cvt8(), table_idx);
                    mov(ptr[rsp + i * sizeof(char)], reg_tmp_64.cvt8());
                    break;
                default:
                    assert(!"unknown src_dt");
            }
        }

        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_val, ptr[rsp]);
                break;
            case memory::bf16:
                uni_vpmovzxwd(vmm_val, ptr[rsp]);
                uni_vpslld(vmm_val, vmm_val, 16);
            break;
            case memory::s8:
                uni_vpmovsxbd(vmm_val, ptr[rsp]);
                break;
            case memory::u8:
                uni_vpmovzxbd(vmm_val, ptr[rsp]);
                break;
            default:
                assert(!"unknown src_dt");
        }
        add(rsp, vlen);
    }

    inline void load_gathered_scalar(Xmm xmm_val, const Xbyak::Address &op, memory::data_type src_dt) {
        uni_vmovdqu(xmm_index, op);
        pack_gathered_scalar(xmm_val, xmm_index, src_dt);
    }

    inline void pack_gathered_scalar(Xmm xmm_val, Xmm xmm_index, memory::data_type src_dt) {
        sub(rsp, vlen);
        uni_vmovdqu(ptr[rsp], xmm_index);
        mov(reg_tmp_64.cvt32(), ptr[rsp]);
        Xbyak::Address table_idx = ptr[reg_src + reg_tmp_64];
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                mov(reg_tmp_64.cvt32(), table_idx);
                mov(ptr[rsp], reg_tmp_64.cvt32());
                break;
            case memory::bf16:
                mov(reg_tmp_64.cvt16(), table_idx);
                mov(ptr[rsp], reg_tmp_64.cvt16());
                break;
            case memory::s8:
            case memory::u8:
                mov(reg_tmp_64.cvt8(), table_idx);
                mov(ptr[rsp], reg_tmp_64.cvt8());
                break;
            default:
                assert(!"unknown src_dt");
        }

        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_val, ptr[rsp]);
                break;
            case memory::bf16:
                pinsrw(xmm_val, ptr[rsp], 0x0);
                uni_vpslld(xmm_val, xmm_val, 16);
                break;
            case memory::s8:
                movsx(reg_tmp_32, ptr[rsp]);
                movq(xmm_val, reg_tmp_64);
                break;
            case memory::u8:
                movzx(reg_tmp_32, ptr[rsp]);
                movq(xmm_val, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }
        add(rsp, vlen);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
                break;
            case memory::bf16:
                uni_vpmovzxwd(vmm_src, op);
                uni_vpslld(vmm_src, vmm_src, 16);
                break;
            case memory::s8:
                uni_vpmovsxbd(vmm_src, op);
                break;
            case memory::u8:
                uni_vpmovzxbd(vmm_src, op);
                break;
            default:
                assert(!"unknown src_dt");
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
                break;
            case memory::bf16:
                pinsrw(xmm_src, op, 0x0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case memory::s8:
                movsx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            case memory::u8:
                movzx(reg_tmp_32, op);
                movq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_dt");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());
        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::bf16:
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(ymm_dst, vmm_dst);
                else
                    emu_vcvtneps2bf16->emit({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case memory::s8:
                if (isa == cpu::avx512_common) {
                    vmaxps(vmm_dst, vmm_zero, vmm_dst);
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case memory::u8:
                if (isa == cpu::avx512_common) {
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != cpu::sse42)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, memory::data_type dst_dt) {
        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
                break;
            case memory::bf16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                pextrw(op, xmm_dst, 0x0);
                break;
            case memory::s8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case memory::u8:
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

MKLDNNShuffleChannelsNode::MKLDNNShuffleChannelsNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNShuffleChannelsNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().empty() || getChildEdges().empty())
        THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() << " gets incorrect number of input/output edges!";

    if (getParentEdgeAt(0)->getDims().ndims() != getChildEdgeAt(0)->getDims().ndims()) {
        THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() << " gets incorrect number of input/output dimensions!";
    }

    auto *layer = getCnnLayer().get();
    dst_dims = layer->outData[0]->getTensorDesc().getDims();
    dims_size = dst_dims.size();

    axis = layer->GetParamAsInt("axis", 1);
    if (axis < 0)
        axis += dims_size;

    if (axis < 0 || axis >= static_cast<int>(dims_size))
        THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() << " gets incorrect input parameters dimensions and axis number!";

    group = layer->GetParamAsUInt("group", 1);
    if (group == 0 || dst_dims[axis] % group)
        THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() <<
                              " gets incorrect group parameter! Group parameter must evenly divide the channel dimension!";

    shuffle_size = dst_dims[axis];
    group_size = dst_dims[axis] / group;
}

void MKLDNNShuffleChannelsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    static const Precision supportedPrecision[] = {
        Precision::FP32,
        Precision::BF16,
        Precision::I32,
        Precision::I8,
        Precision::U8
    };

    Precision inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();
    if (outputPrecision == Precision::BF16 && !mayiuse(avx512_core))
        outputPrecision = Precision::FP32;

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    // shuffle shouldn't change the precision itself
    if (inputDataType != outputDataType)
        inputDataType = outputDataType;

    jit_mode = mayiuse(cpu::sse42) && std::find(std::begin(supportedPrecision), std::end(supportedPrecision), inputPrecision)
                                   != std::end(supportedPrecision);

    data_type = inputDataType;
    data_size = MKLDNNExtensionUtils::sizeOfDataType(data_type);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[0].inPlace = -1;
    config.outConfs[0].inPlace = -1;

    auto pushDesc = [&](memory::format inFormat, memory::format outFormat, memory::data_type dataType, impl_desc_type impl_type) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, inFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, outFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_type, outFormat});
    };

    if (jit_mode) {
        impl_desc_type impl_type = impl_desc_type::jit_sse42;
        if (mayiuse(cpu::avx512_common)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(cpu::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }

        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(0)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(0)->getDims().ndims())), data_type, impl_type);
        if (getParentEdgeAt(0)->getDims().ndims() == 4) {
            if (mayiuse(cpu::avx512_common)) {
                pushDesc(memory::nhwc, memory::nhwc, data_type, impl_type);
                pushDesc(memory::nChw16c, memory::nChw16c, data_type, impl_type);
            } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                pushDesc(memory::nhwc, memory::nhwc, data_type, impl_type);
                pushDesc(memory::nChw8c, memory::nChw8c, data_type, impl_type);
            }
        } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
            if (mayiuse(cpu::avx512_common)) {
                pushDesc(memory::ndhwc, memory::ndhwc, data_type, impl_type);
                pushDesc(memory::nCdhw16c, memory::nCdhw16c, data_type, impl_type);
            } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                pushDesc(memory::ndhwc, memory::ndhwc, data_type, impl_type);
                pushDesc(memory::nCdhw8c, memory::nCdhw8c, data_type, impl_type);
            }
        }
    } else {
        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(0)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(0)->getDims().ndims())), memory::f32, impl_desc_type::ref);
    }
}

void MKLDNNShuffleChannelsNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcDataMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcDataMemPtr || !srcDataMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[0].desc.getLayout();
    if (MKLDNNMemory::GetPlainLayout(getParentEdgeAt(0)->getDims()) == selected_layout) {
        layout = ShuffleChannelsLayoutType::planar_layout;
    } else if ((selected_layout == NHWC) || (selected_layout == NDHWC)) {
        layout = ShuffleChannelsLayoutType::by_channel_layout;
    } else {
        layout = ShuffleChannelsLayoutType::blocked_layout;
    }

    shuffle_innermost = false;
    if ((layout == ShuffleChannelsLayoutType::planar_layout && axis == dims_size - 1) ||
       ((layout == ShuffleChannelsLayoutType::by_channel_layout || layout == ShuffleChannelsLayoutType::blocked_layout) && axis == 1))
        shuffle_innermost = true;

    if (mayiuse(cpu::avx512_common))
        blk_size = 16;
    else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42))
        blk_size = 8;

    permute_mode = false;
    size_t fold = group < group_size ? group : group_size;
    if (mayiuse(cpu::avx2) && shuffle_innermost && (shuffle_size % (fold * blk_size) == 0) &&
       (group == 2 || group == 4))
        permute_mode = true;

    auto jcp = jit_shuffle_channels_config_params();
    jcp.data_type = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[0].desc.getPrecision());
    jcp.data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.data_type);
    jcp.layout = layout;
    jcp.shuffle_innermost = shuffle_innermost;
    jcp.permute_mode = permute_mode;
    jcp.group = group;
    jcp.shuffle_size = shuffle_size;

    if (layout == ShuffleChannelsLayoutType::blocked_layout) {
        jcp.channel_batch = div_up(dst_dims[1], blk_size);
        jcp.shuffle_stride = blk_size;
        for (size_t i = 2; i < dims_size; i++)
            jcp.shuffle_stride *= dst_dims[i];
    }

    if (mayiuse(cpu::avx512_common))
        shuffle_channels_kernel.reset(new jit_uni_shuffle_channels_kernel_f32<cpu::avx512_common>(jcp));
    else if (mayiuse(cpu::avx2))
        shuffle_channels_kernel.reset(new jit_uni_shuffle_channels_kernel_f32<cpu::avx2>(jcp));
    else if (mayiuse(cpu::sse42))
        shuffle_channels_kernel.reset(new jit_uni_shuffle_channels_kernel_f32<cpu::sse42>(jcp));

    jit_mode = jit_mode && shuffle_channels_kernel;
    if (!jit_mode) {
        own_dims[0] = 1;
        for (int i = 0; i < axis; i++)
            own_dims[0] *= dst_dims[i];

        for (size_t i = axis + 1; i < dst_dims.size(); i++)
            dataLength *= dst_dims[i];

        if (dataLength == 0)
            THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() << " gets incorrect input parameters dimension!";

        own_dims[1] = dst_dims[axis] / group;
        own_dims[2] = group;
        ownStrides[0] = dst_dims[axis];
        ownStrides[1] = 1;
        ownStrides[2] = own_dims[1];
        work_amount_dst = ownStrides[0] * own_dims[0];
    }
}

void MKLDNNShuffleChannelsNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData()) +
                   srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
                   MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(srcMemPtr->GetDescriptor().data.data_type));
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData()) +
                   dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
                   MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dstMemPtr->GetDescriptor().data.data_type));

    if (jit_mode) {
        if (layout == ShuffleChannelsLayoutType::planar_layout || layout == ShuffleChannelsLayoutType::by_channel_layout) {
            shuffle_PLN(src_data, dst_data);
        } else {
            shuffle_BLK(src_data, dst_data);
        }
    } else {
        if (layout == ShuffleChannelsLayoutType::planar_layout) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            shuffle_ref(in_ptr, out_ptr);
        } else {
            THROW_IE_EXCEPTION << "ShuffleChannels layer with name " << getName() <<  "only support plain layout on machine w/o sse42.";
        }
    }
}

void MKLDNNShuffleChannelsNode::shuffle_PLN(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t O = 1, I = 1;
    size_t A = dst_dims[axis];
    if (layout == ShuffleChannelsLayoutType::planar_layout) {
        for (int i = 0; i < axis; i++)
            O *= dst_dims[i];
        for (size_t i = axis + 1; i < dst_dims.size(); i++)
            I *= dst_dims[i];
    } else if (layout == ShuffleChannelsLayoutType::by_channel_layout) {
        if (axis != 1) {
            for (int i = 0; i < axis; i++) {
                if (i != 1)
                    O *= dst_dims[i];
            }
            I = dst_dims[1];
            for (size_t i = axis + 1; i < dst_dims.size(); i++)
                if (i != 1)
                    I *= dst_dims[i];
        } else {
            for (int i = 0; i < dims_size; i++) {
                if (i != 1)
                    O *= dst_dims[i];
            }
        }
    }

    if (!shuffle_innermost) {
        parallel_for2d(O, A, [&](size_t o, size_t a) {
            size_t ia = GET_SHUFFLE_INDEX(a);
            size_t work_amount = I;
            GET_PTR(o * A + ia, o * A + a, work_amount);
            shuffle_kernel(in_p, out_p, NULL, NULL, work_amount);
        });
    } else {
        if (permute_mode) {
            parallel_for(O, [&](size_t o) {
                size_t work_amount = A;
                GET_PTR(o, o, work_amount);
                if (mayiuse(cpu::avx512_common)) {
                    if (group == 2)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx512_group_2, work_amount);
                    else if (group == 4)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx512_group_4, work_amount);
                } else if (mayiuse(cpu::avx2)) {
                    if (group == 2)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx2_group_2, work_amount);
                    else if (group == 4)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx2_group_4, work_amount);
                }
            });
        } else {
            parallel_for(O, [&](size_t o) {
                std::vector<int> index_buffer(A);
                for (size_t a = 0; a < A; a++) {
                    index_buffer[a] = GET_SHUFFLE_INDEX(a) * data_size;
                }
                size_t work_amount = A;
                GET_PTR(o, o, work_amount);
                shuffle_kernel(in_p, out_p, static_cast<int *>(&index_buffer[0]), NULL, work_amount);
            });
        }
    }
}

void MKLDNNShuffleChannelsNode::shuffle_BLK(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t N = dst_dims[0];
    size_t CB = div_up(dst_dims[1], blk_size);
    size_t D = 1;
    if (dims_size == 5)
        D = dst_dims[2];
    size_t H = dst_dims[dims_size - 2];
    size_t W = dst_dims[dims_size - 1];

    if (!shuffle_innermost) {
        if (axis == 0) { //shuffle N
            parallel_for(N, [&](size_t n) {
                size_t in = GET_SHUFFLE_INDEX(n);
                size_t work_amount = CB * D * H * W * blk_size;
                GET_PTR(in, n, work_amount);
                shuffle_kernel(in_p, out_p, NULL, NULL, work_amount);
            });
        } else if (dims_size == 5 && axis == 2) { //shuffle D
            parallel_for3d(N, CB, D, [&](size_t n, size_t cb, size_t d) {
                size_t id = GET_SHUFFLE_INDEX(d);
                size_t work_amount = H * W * blk_size;
                GET_PTR((n * CB + cb) * D + id, (n * CB + cb) * D + d, work_amount);
                shuffle_kernel(in_p, out_p, NULL, NULL, work_amount);
            });
        } else if (axis == dims_size - 2) { //shuffle H
            parallel_for4d(N, CB, D, H, [&](size_t n, size_t cb, size_t d, size_t h) {
                size_t ih = GET_SHUFFLE_INDEX(h);
                size_t work_amount = W * blk_size;
                GET_PTR(((n * CB + cb) * D + d) * H + ih, ((n * CB + cb) * D + d) * H + h, work_amount);
                shuffle_kernel(in_p, out_p, NULL, NULL, work_amount);
            });
        } else if (axis == dims_size - 1) { //shuffle W
            parallel_for5d(N, CB, D, H, W, [&](size_t n, size_t cb, size_t d, size_t h, size_t w) {
                size_t iw = GET_SHUFFLE_INDEX(w);
                size_t work_amount = blk_size;
                GET_PTR((((n * CB + cb) * D + d) * H + h) * W + iw, (((n * CB + cb) * D + d) * H + h) * W + w, work_amount);
                shuffle_kernel(in_p, out_p, NULL, NULL, work_amount);
            });
        }
    } else { //shuffle C
        if (permute_mode) {
            size_t O = N;
            size_t A = CB * D * H * W * blk_size;
            parallel_for(O, [&](size_t o) {
                size_t work_amount = A;
                GET_PTR(o, o, work_amount);
                if (mayiuse(cpu::avx512_common)) {
                    if (group == 2)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx512_group_2, work_amount);
                    else if (group == 4)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx512_group_4, work_amount);
                } else if (mayiuse(cpu::avx2)) {
                    if (group == 2)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx2_group_2, work_amount);
                    else if (group == 4)
                        shuffle_kernel(in_p, out_p, NULL, tab_avx2_group_4, work_amount);
                }
            });
        } else {
            parallel_for5d(N, CB, D, H, W, [&](size_t n, size_t cb, size_t d, size_t h, size_t w) {
                std::vector<int> index_buffer(blk_size);
                for (size_t blk_k = 0; blk_k < blk_size; blk_k++) {
                    size_t blk_ik = blk_k, icb = cb;
                    if (blk_k < shuffle_size) {
                        size_t k = cb * blk_size + blk_k;
                        size_t ik = GET_SHUFFLE_INDEX(k);
                        icb = ik / blk_size;
                        blk_ik = ik % blk_size;
                    }
                    index_buffer[blk_k] = ((((icb * D + d) * H + h) * W + w) * blk_size + blk_ik) * data_size;
                }
                size_t work_amount = blk_size;
                GET_PTR(n * CB * D * H * W, (((n * CB + cb) * D + d) * H + h) * W + w, work_amount);
                shuffle_kernel(in_p, out_p, static_cast<int *>(&index_buffer[0]), NULL, work_amount);
            });
        }
    }
}

inline void MKLDNNShuffleChannelsNode::shuffle_kernel(const uint8_t *in_p, uint8_t *out_p,
                                                      const int *src_idx, const int *tab_idx, size_t work_amount) {
    auto arg = jit_shuffle_channels_call_args();
    arg.src = static_cast<const void *>(in_p);
    arg.dst = static_cast<void *>(out_p);
    arg.index = src_idx;
    arg.tab_idx = tab_idx;
    arg.work_amount = work_amount;
    (*shuffle_channels_kernel)(&arg);
}

void MKLDNNShuffleChannelsNode::shuffle_ref(const float *in_ptr, float *out_ptr) {
    if (dataLength > 1) {
        // Vectorized & Parallel
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0, src_idx = 0;
            size_t counters[CNTR_SIZE] = { 0 };
            splitter(work_amount_dst, nthr, ithr, start, end);
            src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
            for (size_t iwork = start, dst_idx = start * dataLength; iwork < end; ++iwork, dst_idx += dataLength) {
                memcpy(&out_ptr[dst_idx], &in_ptr[dataLength * src_idx], sizeof(float) * dataLength);
                src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
            }
        });
    } else {
        // Parallel
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0, src_idx = 0;
            size_t counters[CNTR_SIZE] = { 0 };
            splitter(work_amount_dst, nthr, ithr, start, end);
            src_idx = initter(start, CNTR_SIZE, counters, own_dims, ownStrides);
            for (size_t iwork = start; iwork < end; ++iwork) {
                out_ptr[iwork] = in_ptr[src_idx];
                src_idx = updater(src_idx, CNTR_SIZE, counters, own_dims, ownStrides);
            }
        });
    }
}

inline size_t MKLDNNShuffleChannelsNode::initter(size_t start, size_t size, size_t *counters, size_t *own_dims, size_t *ownStrides) {
    size_t i = start;
    size_t idx = 0;
    for (int j = size - 1; j >= 0; j--) {
        counters[j] = i % own_dims[j];
        idx += counters[j] * ownStrides[j];
        i /= own_dims[j];
    }
    return idx;
}

inline size_t MKLDNNShuffleChannelsNode::updater(size_t idx, size_t size, size_t *counters, size_t *own_dims, size_t *ownStrides) {
    size_t i = 1;
    for (int j = size - 1; j >= 0; j--) {
        counters[j]++;
        if (counters[j] < own_dims[j]) {
            idx += ownStrides[j];
            break;
        } else {
            counters[j] = 0;
            i = 0;
        }
    }
    if (!i) {
        for (idx = 0; i < CNTR_SIZE; ++i)
            idx += counters[i] * ownStrides[i];
    }
    return idx;
}

bool MKLDNNShuffleChannelsNode::created() const {
    return getType() == ShuffleChannels;
}

REG_MKLDNN_PRIM_FOR(MKLDNNShuffleChannelsNode, ShuffleChannels);
