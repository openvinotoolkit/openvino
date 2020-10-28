// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reduce_node.h"
#include "desc_iterator.hpp"
#include "mkldnn_quantize_node.h"
#include <legacy/ie_layers.h>
#include <mkldnn.hpp>
#include <string>
#include <vector>
#include <set>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include <algorithm>

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"
#include "jit_uni_quantization.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

#define SET_SRC_DIM_VALUE(batch, channel, depth, height, width) IB = batch;   \
                                                                IC = channel; \
                                                                ID = depth;   \
                                                                IH = height;  \
                                                                IW = width;
#define SET_DST_DIM_VALUE(batch, channel, depth, height, width) OB = batch;   \
                                                                OC = channel; \
                                                                OD = depth;   \
                                                                OH = height;  \
                                                                OW = width;

#define GET_OFF(field) offsetof(jit_reduce_call_args, field)

#define GET_PTR_N_PLN              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * IC * ID * IH * IW;               \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OC * OD * OH * OW;
#define GET_PTR_NC_PLN             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * ic * ID * IH * IW;                    \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * oc * OD * OH * OW;
#define GET_PTR_NCD_PLN            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW;                         \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW;
#define GET_PTR_NCDH_PLN           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW;                              \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW;
#define GET_PTR_NCD_BASE_PTR_N_PLN const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (ic * ID + id) * IH * IW;             \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (oc * OD + od) * OH * OW;
#define GET_PTR_N_BLK              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * ICB * ID * IH * IW * blk_size;   \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OCB * OD * OH * OW * blk_size;
#define GET_PTR_NC_BLK             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * icb * ID * IH * IW * blk_size;        \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * ocb * OD * OH * OW * blk_size;
#define GET_PTR_NCD_BLK            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW * blk_size;              \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW * blk_size;
#define GET_PTR_NCDH_BLK           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW * blk_size;                   \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW * blk_size;
#define GET_PTR_NCDHW_BLK          const uint8_t    *in_ptr_ncdhw  = in_ptr_ncdh  + src_data_size * iw * blk_size;                        \
                                         uint8_t    *out_ptr_ncdhw = out_ptr_ncdh + dst_data_size * ow * blk_size;
#define GET_PTR_NCD_BASE_PTR_N_BLK const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (icb * ID + id) * IH * IW * blk_size; \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (ocb * OD + od) * OH * OW * blk_size;

template <cpu_isa_t isa>
struct jit_uni_reduce_kernel_f32 : public jit_uni_reduce_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduce_kernel_f32)

    explicit jit_uni_reduce_kernel_f32(jit_reduce_config_params jcp)
    : jit_uni_reduce_kernel(jcp), jit_generator() {
        exp_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, alg_kind::eltwise_exp, 0.f, 0.f));

        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        if (jcp_.planar_layout)
            mov(reg_reduce_w, ptr[reg_params + GET_OFF(reduce_w)]);

        if (jcp_.reduce_mode == Reduce::And || jcp_.reduce_mode == Reduce::L1 || jcp_.reduce_mode == Reduce::Max ||
            jcp_.reduce_mode == Reduce::Min || jcp_.reduce_mode == Reduce::Prod || jcp_.reduce_mode == Reduce::Or) {
            mov(reg_table, l_table);
        }

        if (isa == cpu::avx512_common || jcp_.reduce_mode == Reduce::And || jcp_.reduce_mode == Reduce::Or)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        if ((isa == cpu::avx512_common && jcp_.reduce_mode == Reduce::And) || jcp_.reduce_mode == Reduce::Or) {
            uni_vmovups(vmm_aux, table_val(0));
        }

        reduce_main();
        reduce_tail();

        this->postamble();

        if (jcp_.reduce_mode == Reduce::And || jcp_.reduce_mode == Reduce::L1 || jcp_.reduce_mode == Reduce::Max ||
            jcp_.reduce_mode == Reduce::Min || jcp_.reduce_mode == Reduce::Prod || jcp_.reduce_mode == Reduce::Or) {
            prepare_aux_table();
        } else if (jcp_.reduce_mode == Reduce::LogSumExp) {
            exp_injector->prepare_table();
        }

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Address table_val(int index) { return ptr[reg_table + index * vlen]; }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_dst = r9;
    Xbyak::Reg64 reg_work_amount = r10;
    Xbyak::Reg64 reg_reduce_w = r11;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg8 reg_tmp_8 = r13b;
    Xbyak::Reg32 reg_tmp_32 = r13d;
    Xbyak::Reg64 reg_tmp_64 = r13;

    Vmm vmm_aux = Vmm(0);
    Xmm xmm_aux = Xmm(0);
    Vmm vmm_src = Vmm(1);
    Xmm xmm_src = Xmm(1);
    Vmm vmm_dst = Vmm(2);
    Xmm xmm_dst = Xmm(2);
    Vmm vmm_zero = Vmm(3);
    Xmm xmm_zero = Xmm(3);
    Vmm vmm_dst_aux = Vmm(4);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(5);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(6);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(7);

    const Xbyak::Opmask k_mask = Xbyak::Opmask(1);

    Xbyak::Label l_table;

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector;

    inline void reduce_main() {
        // ================================================================
        // ***isa: AVX512***
        // Reduce::And (Logical And)
        // step 1: init dst 0x3f800000 (1.0f)
        //              aux 0x3f800000 (1.0f)
        //             zero 0x00000000 (0.0f)
        // step 2: if src equals 0, set mask bit 0, else set mask bit 1
        // step 3: src = mask bit == 0 ? zero : aux
        // step 4: dst = dst & src
        //                  src    mask_bit    new_src    dst    new_dst
        //         case 1    ~0        1         1.0f     1.0f     1.0f
        //         case 2     0        0         0.0f     1.0f     0.0f
        //         case 3    ~0        1         1.0f     0.0f     0.0f
        //         case 4     0        0         0.0f     0.0f     0.0f
        // step 5: loop: offset src, and do step 2 and step 3
        //
        // Reduce::Or (Logical Or)
        // step 1: init dst 0x00000000 (0.0f)
        //              aux 0x3f800000 (1.0f)
        //             zero 0x00000000 (0.0f)
        // step 2: if src equals 0, set mask bit 0, else set mask bit 1
        // step 3: src = mask bit == 0 ? zero : aux
        // step 4: dst = dst | src
        //                  src    mask_bit    new_src    dst    new_dst
        //         case 1     0        0         0.0f     0.0f     0.0f
        //         case 2    ~0        1         1.0f     0.0f     1.0f
        //         case 3     0        0         0.0f     1.0f     1.0f
        //         case 4    ~0        1         1.0f     1.0f     1.0f
        // step 5: loop: offset src, and do step 2 and step 3
        // ================================================================
        // ***isa: OTHER***
        // Reduce::And (Logical And)
        // step 1: init dst 0x3f800000 (1.0f)
        // step 2: if src equals 0, set it 0x00000000, else set 0xffffffff
        // step 3: dst = dst & src
        //         0x3f800000 = 0x3f800000 & 0xffffffff (result: 1.0f)
        //         0x00000000 = 0x3f800000 & 0x00000000 (result: 0.0f)
        //         0x00000000 = 0x00000000 & 0xffffffff (result: 0.0f)
        //         0x00000000 = 0x00000000 & 0x00000000 (result: 0.0f)
        // step 4: loop: offset src, and do step 2 and step 3
        //
        // Reduce::Or (Logical Or)
        // step 1: init dst 0x00000000 (0.0f)
        //              aux 0x3f800000 (1.0f)
        // step 2: dst = dst | src
        //         0x00000000 = 0x00000000 | 0x00000000
        //                  A = 0x00000000 | A
        //                  A =          A | 0x00000000
        //                  C =          A | B
        // (A, B stand for number other than 0x00000000)
        // step 3: loop: offset src, and do step 2
        // step 4: if dst equals 0, set it 0x00000000, else set 0xffffffff
        // step 5: dst = dst & aux
        //         0x00000000 = 0x00000000 & 0x3f800000 (result: 0.0f)
        //         0x3f800000 = 0xffffffff & 0x3f800000 (result: 1.0f)
        // ================================================================
        Xbyak::Label reduce_to_vector_label;
        Xbyak::Label reduce_to_scalar_label;
        Xbyak::Label reduce_main_end_label;
        if (jcp_.planar_layout) {
            cmp(reg_reduce_w, 1); // planar layout reducing W
            je(reduce_to_scalar_label, T_NEAR);
        }

        // store vmm_dst directly into memory after reducing
        // cases: [planar layout reducing other dimensions but W] [blocked layout]
        L(reduce_to_vector_label);
        {
            int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
            cmp(reg_work_amount, step);
            jl(reduce_main_end_label, T_NEAR); //avoid illegal loading and storing

            if (jcp_.reduce_mode == Reduce::L1) {
                uni_vmovups(vmm_aux, table_val(1));
            }

            // load
            load_dst_vector();

            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            // reduce
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                load_vector(vmm_src, ptr[reg_src], jcp_.src_dt);
                reduce_kernel(vmm_src, vmm_dst);

                if (isa == cpu::sse42) {
                    load_vector(vmm_src, ptr[reg_src + 4 * jcp_.src_data_size], jcp_.src_dt);
                    reduce_kernel(vmm_src, vmm_dst_aux);
                }

                add(reg_src, step * jcp_.src_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            // store
            store_dst_vector();

            jmp(reduce_main_end_label, T_NEAR);
        }

        // reduce vector in vmm_dst to be a scalar before store into memory
        // cases: [planar layout reducing W]
        L(reduce_to_scalar_label);
        {
            // init dst, dst loading is embedded in horiz_reduce_store
            switch (jcp_.reduce_mode) {
                case Reduce::And:
                case Reduce::Prod:
                    uni_vmovups(vmm_dst, table_val(0));
                    break;
                case Reduce::L1:
                    uni_vmovups(vmm_aux, table_val(1));
                    uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                    break;
                case Reduce::L2:
                case Reduce::LogSum:
                case Reduce::LogSumExp:
                case Reduce::Mean:
                case Reduce::Or:
                case Reduce::Sum:
                case Reduce::SumSquare:
                    uni_vpxor(vmm_dst, vmm_dst, vmm_dst);
                    break;
                case Reduce::Max:
                    if (jcp_.dst_dt == memory::f32)
                        uni_vmovups(vmm_dst, table_val(2));
                    else
                        uni_vmovups(vmm_dst, table_val(4));
                    break;
                case Reduce::Min:
                    if (jcp_.dst_dt == memory::f32)
                        uni_vmovups(vmm_dst, table_val(3));
                    else
                        uni_vmovups(vmm_dst, table_val(5));
                    break;
                default:
                    assert(!"unsupported reduce mode");
            }
            // reduce
            reduce_main_loop();
            if (jcp_.reduce_mode == Reduce::Or && isa != avx512_common) {
                vcmpneqps(vmm_dst, vmm_dst, vmm_zero);
                uni_vandps(vmm_dst, vmm_dst, vmm_aux);
            }
            // store
            // store after horizontal calculation and calculation with loaded original ptr[reg_dst]
            load_embedded_horiz_reduce_store(vmm_dst, jcp_.dst_dt);
        }

        L(reduce_main_end_label);
    }

    inline void reduce_tail() {
        if (jcp_.reduce_mode == Reduce::L1) {
            uni_vmovups(xmm_aux, table_val(1));
        }

        Xbyak::Label tail_dst_shifted_label;
        Xbyak::Label tail_dst_fixed_label;
        Xbyak::Label reduce_tail_end_label;
        if (jcp_.planar_layout) {
            cmp(reg_reduce_w, 1);  // planar layout reducing W
            je(tail_dst_fixed_label, T_NEAR);
        }

        // each src scalar reduce to each dst scalar (X1, X2, X3, ...) -> (Y1, Y2, Y3, ...)
        // cases: [planar layout reducing other dimensions but W] [blocked layout concern padding]
        L(tail_dst_shifted_label);
        {
            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);
                load_scalar(xmm_src, ptr[reg_src], jcp_.src_dt);

                // reduce
                reduce_kernel_scalar(xmm_src, xmm_dst);
                if (jcp_.reduce_mode == Reduce::Or) {
                    vcmpneqps(xmm_dst, xmm_dst, xmm_zero);
                    uni_vandps(xmm_dst, xmm_dst, xmm_aux);
                }

                // store
                store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                add(reg_src, step * jcp_.src_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            jmp(reduce_tail_end_label, T_NEAR);
        }

        // each src scalar reduce to the same dst scalar (X1, X2, X3, ...) -> (Y1)
        // cases: [planar layout reducing W]
        L(tail_dst_fixed_label);
        {
            // load
            load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            // reduce
            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                load_scalar(xmm_src, ptr[reg_src], jcp_.src_dt);

                reduce_kernel_scalar(xmm_src, xmm_dst);
                if (jcp_.reduce_mode == Reduce::Or) {
                    vcmpneqps(xmm_dst, xmm_dst, xmm_zero);
                    uni_vandps(xmm_dst, xmm_dst, xmm_aux);
                }

                add(reg_src, step * jcp_.src_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            // store
            store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);
            add(reg_dst, step * jcp_.dst_data_size);
        }

        L(reduce_tail_end_label);
    }

    inline void reduce_main_loop() {
        Xbyak::Label reduce_loop_label;
        Xbyak::Label reduce_loop_end_label;

        int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
        L(reduce_loop_label);
        {
            cmp(reg_work_amount, step);
            jl(reduce_loop_end_label, T_NEAR);

            load_vector(vmm_src, ptr[reg_src], jcp_.src_dt);
            reduce_kernel(vmm_src, vmm_dst);

            if (isa == cpu::sse42) {
                load_vector(vmm_src, ptr[reg_src + 4 * jcp_.src_data_size], jcp_.src_dt);
                reduce_kernel(vmm_src, vmm_dst);
            }

            add(reg_src, step * jcp_.src_data_size);
            sub(reg_work_amount, step);

            jmp(reduce_loop_label, T_NEAR);
        }
        L(reduce_loop_end_label);
    }

    inline void reduce_kernel(Vmm vmm_src, Vmm vmm_dst) {
        switch (jcp_.reduce_mode) {
            case Reduce::And:
                if (isa == avx512_common) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, vmm_zero, vmm_aux);
                } else {
                    vcmpneqps(vmm_src, vmm_src, vmm_zero);
                }
                uni_vandps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::L1:
                uni_vandps(vmm_src, vmm_src, vmm_aux);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::LogSum:
            case Reduce::Mean:
            case Reduce::Sum:
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::Max:
                uni_vmaxps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::Min:
                uni_vminps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::L2:
            case Reduce::SumSquare:
                uni_vmulps(vmm_src, vmm_src, vmm_src);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::LogSumExp:
                exp_injector->compute_vector_range(vmm_src.getIdx(), vmm_src.getIdx() + 1);
                uni_vaddps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::Or:
                if (isa == avx512_common) {
                    vcmpps(k_mask, vmm_src, vmm_zero, _cmp_neq_uq);
                    vblendmps(vmm_src | k_mask, vmm_zero, vmm_aux);
                }
                uni_vorps(vmm_dst, vmm_dst, vmm_src);
                break;
            case Reduce::Prod:
                uni_vmulps(vmm_dst, vmm_dst, vmm_src);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    inline void reduce_kernel_scalar(Xmm xmm_src, Xmm xmm_dst) {
        switch (jcp_.reduce_mode) {
            case Reduce::And:
                vcmpneqps(xmm_src, xmm_src, xmm_zero);
                uni_vandps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::L1:
                uni_vandps(xmm_src, xmm_src, xmm_aux);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::LogSum:
            case Reduce::Mean:
            case Reduce::Sum:
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::Max:
                uni_vmaxps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::Min:
                uni_vminps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::L2:
            case Reduce::SumSquare:
                uni_vmulps(xmm_src, xmm_src, xmm_src);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::LogSumExp:
                exp_injector->compute_vector_range(xmm_src.getIdx(), xmm_src.getIdx() + 1);
                uni_vaddps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::Or:
                uni_vorps(xmm_dst, xmm_dst, xmm_src);
                break;
            case Reduce::Prod:
                uni_vmulps(xmm_dst, xmm_dst, xmm_src);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    inline void load_dst_vector() {
        load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
        if (isa == cpu::sse42)
            load_vector(vmm_dst_aux, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);
    }

    inline void store_dst_vector() {
        if (jcp_.reduce_mode == Reduce::Or && isa != avx512_common) {
            vcmpneqps(vmm_dst, vmm_dst, vmm_zero);
            uni_vandps(vmm_dst, vmm_dst, vmm_aux);
            if (isa == cpu::sse42) {
                vcmpneqps(vmm_dst_aux, vmm_dst_aux, vmm_zero);
                uni_vandps(vmm_dst_aux, vmm_dst_aux, vmm_aux);
            }
        }
        store_vector(ptr[reg_dst], vmm_dst, jcp_.dst_dt);
        if (isa == cpu::sse42)
            store_vector(ptr[reg_dst + 4 * jcp_.dst_data_size], vmm_dst_aux, jcp_.dst_dt);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
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

        if (src_dt != memory::f32)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
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

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        if (dst_dt != memory::f32) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::s8:
                if (isa == avx512_common) {
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
                if (isa == avx512_common) {
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
        if (dst_dt != memory::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
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

    inline void load_embedded_horiz_reduce_store(Vmm vmm_dst, memory::data_type dst_dt) {
        if (isa == cpu::sse42) {
            load_embedded_horiz_store(vmm_dst, dst_dt);
        } else if (isa == cpu::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_aux1, ymm_dst, 0);
            vextractf128(xmm_aux2, ymm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            load_embedded_horiz_store(xmm_aux1, dst_dt);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_aux1, zmm_dst, 0);
            vextractf32x4(xmm_aux2, zmm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_dst, 2);
            vextractf32x4(xmm_aux3, zmm_dst, 3);
            horiz_ps(xmm_aux2, xmm_aux3);
            horiz_ps(xmm_aux1, xmm_aux2);
            load_embedded_horiz_store(xmm_aux1, dst_dt);
        }
    }

    inline void load_embedded_horiz_store(Xbyak::Xmm xmm_dst, memory::data_type dst_dt) {
        movshdup(xmm_aux3, xmm_dst); // dst:1,2,3,4; aux3:2,2,4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        movhlps(xmm_aux3, xmm_dst);  // aux3:f(3,4),f(4,4),4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2,3,4),...
        switch (dst_dt) {
            case memory::f32:
                movss(xmm_aux3, ptr[reg_dst]);
                horiz_ps(xmm_dst, xmm_aux3);
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::s32:
                movss(xmm_aux3, ptr[reg_dst]);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::u8:
                vpbroadcastb(xmm_aux3, ptr[reg_dst]);
                uni_vpmovzxbd(xmm_aux3, xmm_aux3);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            case memory::s8:
                vpbroadcastb(xmm_aux3, ptr[reg_dst]);
                uni_vpmovsxbd(xmm_aux3, xmm_aux3);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void horiz_ps(const Xmm& xmm, const Operand& op) {
        switch (jcp_.reduce_mode) {
            case Reduce::And:
                andps(xmm, op);
                break;
            case Reduce::L1:
            case Reduce::L2:
            case Reduce::LogSum:
            case Reduce::Mean:
            case Reduce::Sum:
            case Reduce::SumSquare:
            case Reduce::LogSumExp:
                addps(xmm, op);
                break;
            case Reduce::Max:
                maxps(xmm, op);
                break;
            case Reduce::Min:
                minps(xmm, op);
                break;
            case Reduce::Or:
                orps(xmm, op);
                break;
            case Reduce::Prod:
                mulps(xmm, op);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }

    void prepare_aux_table() {
        auto broadcast_int = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(float); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table);

        broadcast_int(aux_vals.float_one);
        broadcast_int(aux_vals.float_abs);
        broadcast_int(aux_vals.float_min);
        broadcast_int(aux_vals.float_max);
        broadcast_int(aux_vals.int32_min);
        broadcast_int(aux_vals.int32_max);
    }

    const struct aux_vals_type {
        int float_one = 0x3f800000; // 1.0f
        int float_abs = 0x7fffffff; // mask to make positive
        int float_min = 0xff7fffff; // float minimum
        int float_max = 0x7f7fffff; // float maximum
        int int32_min = 0xcf000000; // -2^31 presented in float
        int int32_max = 0x4effffff; // 2^31-1 presented in float
    } aux_vals;
};

template <cpu_isa_t isa>
struct jit_uni_reduce_post_kernel_f32 : public jit_uni_reduce_post_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduce_post_kernel_f32)

    explicit jit_uni_reduce_post_kernel_f32(jit_reduce_config_params jcp)
    : jit_uni_reduce_post_kernel(jcp), jit_generator() {
        log_injector.reset(new jit_uni_eltwise_injector_f32<isa>(this, alg_kind::eltwise_log, 0.f, 0.f));

        this->preamble();

        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_divisor, ptr[reg_params + GET_OFF(divisor)]);
        if (!jcp_.planar_layout)
            mov(reg_reduce_c, ptr[reg_params + GET_OFF(reduce_c)]);

        if (isa == cpu::avx512_common)
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        reduce_post_main();
        if (jcp_.planar_layout)
            reduce_post_tail();

        this->postamble();

        if (jcp_.reduce_mode == Reduce::LogSum || jcp_.reduce_mode == Reduce::LogSumExp) {
            log_injector->prepare_table();
        }

        ker_ = (decltype(ker_)) this->getCode();
    }

private:
    using Vmm = typename conditional3<isa == cpu::sse42, Xbyak::Xmm, isa == cpu::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;
    size_t vlen = cpu_isa_traits<isa>::vlen;

    Xbyak::Reg64 reg_dst = r8;
    Xbyak::Reg64 reg_work_amount = r9;
    Xbyak::Reg64 reg_divisor = r10;
    Xbyak::Reg64 reg_reduce_c = r11;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg8 reg_tmp_8 = r12b;
    Xbyak::Reg32 reg_tmp_32 = r12d;
    Xbyak::Reg64 reg_tmp_64 = r12;

    Vmm vmm_aux = Vmm(0);
    Xmm xmm_aux = Xmm(0);
    Vmm vmm_dst = Vmm(1);
    Xmm xmm_dst = Xmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_dst_aux = Vmm(3);
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(5);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(6);

    std::shared_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector;

    inline void reduce_post_main() {
        Xbyak::Label reduce_channel_label;
        Xbyak::Label reduce_map_label;
        if (jcp_.planar_layout) {
            jmp(reduce_map_label, T_NEAR);
        } else {
            cmp(reg_reduce_c, 1);
            jne(reduce_map_label, T_NEAR);
        }

        // further reduce channel block since reduce channel batch has already been reduced
        // (X1, X2, X3, X4, X5, X6, X7, X8) -> (Y1, N/A, N/A, N/A, N/A, N/A, N/A, N/A)
        // cases: [blocked layout reducing channel dimensions]
        L(reduce_channel_label);
        {
            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
                if (isa == cpu::sse42)
                    load_vector(vmm_dst_aux, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);

                // reduce and store
                horiz_reduce_store(vmm_dst, jcp_.dst_dt);
                if (isa == cpu::sse42)
                    load_embedded_horiz_reduce_store(vmm_dst_aux, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);

            mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
            mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        }

        // reduce map for value in dst memory
        // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean]
        L(reduce_map_label);
        {
            if (jcp_.reduce_mode == Reduce::L2 || jcp_.reduce_mode == Reduce::Mean ||
                jcp_.reduce_mode == Reduce::LogSum || jcp_.reduce_mode == Reduce::LogSumExp) {
                if (jcp_.reduce_mode == Reduce::Mean)
                    uni_vbroadcastss(vmm_aux, ptr[reg_divisor]);

                Xbyak::Label reduce_loop_label;
                Xbyak::Label reduce_loop_end_label;

                int step = vlen / sizeof(float) < 8 ? 8 : vlen / sizeof(float);
                L(reduce_loop_label);
                {
                    cmp(reg_work_amount, step);
                    jl(reduce_loop_end_label, T_NEAR);

                    // load
                    load_vector(vmm_dst, ptr[reg_dst], jcp_.dst_dt);
                    if (isa == cpu::sse42)
                        load_vector(vmm_dst_aux, ptr[reg_dst + 4 * jcp_.dst_data_size], jcp_.dst_dt);

                    // reduce
                    reduce_map_kernel(vmm_dst);
                    if (isa == cpu::sse42)
                        reduce_map_kernel(vmm_dst_aux);

                    // store
                    store_vector(ptr[reg_dst], vmm_dst, jcp_.dst_dt);
                    if (isa == cpu::sse42)
                        store_vector(ptr[reg_dst + 4 * jcp_.dst_data_size], vmm_dst_aux, jcp_.dst_dt);

                    add(reg_dst, step * jcp_.dst_data_size);
                    sub(reg_work_amount, step);

                    jmp(reduce_loop_label, T_NEAR);
                }
                L(reduce_loop_end_label);
            }
        }
    }

    inline void reduce_post_tail() {
        // reduce map for tail in dst memory
        // cases: [ReduceL2] [ReduceLogSum] [ReduceLogSumExp] [ReduceMean] in planar layout
        if (jcp_.reduce_mode == Reduce::L2 || jcp_.reduce_mode == Reduce::Mean ||
                jcp_.reduce_mode == Reduce::LogSum || jcp_.reduce_mode == Reduce::LogSumExp) {
            if (jcp_.reduce_mode == Reduce::Mean)
                uni_vbroadcastss(xmm_aux, ptr[reg_divisor]);

            Xbyak::Label reduce_loop_label;
            Xbyak::Label reduce_loop_end_label;

            int step = 1;
            L(reduce_loop_label);
            {
                cmp(reg_work_amount, step);
                jl(reduce_loop_end_label, T_NEAR);

                // load
                load_scalar(xmm_dst, ptr[reg_dst], jcp_.dst_dt);

                // reduce
                reduce_map_kernel_scalar(xmm_dst);

                // store
                store_scalar(ptr[reg_dst], xmm_dst, jcp_.dst_dt);

                add(reg_dst, step * jcp_.dst_data_size);
                sub(reg_work_amount, step);

                jmp(reduce_loop_label, T_NEAR);
            }
            L(reduce_loop_end_label);
        }
    }

    inline void reduce_map_kernel(Vmm vmm_dst) {
        if (jcp_.reduce_mode == Reduce::Mean)
            uni_vdivps(vmm_dst, vmm_dst, vmm_aux);
        else if (jcp_.reduce_mode == Reduce::L2)
            uni_vsqrtps(vmm_dst, vmm_dst);
        else if (jcp_.reduce_mode == Reduce::LogSum || jcp_.reduce_mode == Reduce::LogSumExp)
            log_injector->compute_vector_range(vmm_dst.getIdx(), vmm_dst.getIdx() + 1);
    }

    inline void reduce_map_kernel_scalar(Xmm xmm_dst) {
        if (jcp_.reduce_mode == Reduce::Mean)
            uni_vdivps(xmm_dst, xmm_dst, xmm_aux);
        else if (jcp_.reduce_mode == Reduce::L2)
            uni_vsqrtps(xmm_dst, xmm_dst);
        else if (jcp_.reduce_mode == Reduce::LogSum || jcp_.reduce_mode == Reduce::LogSumExp)
            log_injector->compute_vector_range(xmm_dst.getIdx(), xmm_dst.getIdx() + 1);
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(vmm_src, op);
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

        if (src_dt != memory::f32)
            uni_vcvtdq2ps(vmm_src, vmm_src);
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, memory::data_type src_dt) {
        switch (src_dt) {
            case memory::f32:
            case memory::s32:
                movss(xmm_src, op);
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

        if (src_dt != data_type::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, memory::data_type dst_dt) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        if (dst_dt != memory::f32) {
            uni_vcvtps2dq(vmm_dst, vmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                uni_vmovups(op, vmm_dst);
                break;
            case memory::s8:
                if (isa == avx512_common) {
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
                if (isa == avx512_common) {
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
        if (dst_dt != memory::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_dt) {
            case memory::f32:
            case memory::s32:
                movss(op, xmm_dst);
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

    inline void horiz_reduce_store(Vmm vmm_dst, memory::data_type dst_dt) {
        if (isa == cpu::sse42) {
            horize_store(vmm_dst, dst_dt);
        } else if (isa == cpu::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_aux1, ymm_dst, 0);
            vextractf128(xmm_aux2, ymm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            horize_store(xmm_aux1, dst_dt);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_aux1, zmm_dst, 0);
            vextractf32x4(xmm_aux2, zmm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_dst, 2);
            vextractf32x4(xmm_aux3, zmm_dst, 3);
            horiz_ps(xmm_aux2, xmm_aux3);
            horiz_ps(xmm_aux1, xmm_aux2);
            horize_store(xmm_aux1, dst_dt);
        }
    }

    inline void horize_store(Xbyak::Xmm xmm_dst, memory::data_type dst_dt) {
        movshdup(xmm_aux3, xmm_dst); // dst:1,2,3,4; aux3:2,2,4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        movhlps(xmm_aux3, xmm_dst);  // aux3:f(3,4),f(4,4),4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2,3,4),...
        switch (dst_dt) {
            case memory::f32:
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::s32:
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::u8:
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            case memory::s8:
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void load_embedded_horiz_reduce_store(Vmm vmm_dst, memory::data_type dst_dt) {
        if (isa == cpu::sse42) {
            load_embedded_horiz_store(vmm_dst, dst_dt);
        } else if (isa == cpu::avx2) {
            Xbyak::Ymm ymm_dst = Xbyak::Ymm(vmm_dst.getIdx());
            vextractf128(xmm_aux1, ymm_dst, 0);
            vextractf128(xmm_aux2, ymm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            load_embedded_horiz_store(xmm_aux1, dst_dt);
        } else {
            Xbyak::Zmm zmm_dst = Xbyak::Zmm(vmm_dst.getIdx());
            vextractf32x4(xmm_aux1, zmm_dst, 0);
            vextractf32x4(xmm_aux2, zmm_dst, 1);
            horiz_ps(xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_dst, 2);
            vextractf32x4(xmm_aux3, zmm_dst, 3);
            horiz_ps(xmm_aux2, xmm_aux3);
            horiz_ps(xmm_aux1, xmm_aux2);
            load_embedded_horiz_store(xmm_aux1, dst_dt);
        }
    }

    inline void load_embedded_horiz_store(Xbyak::Xmm xmm_dst, memory::data_type dst_dt) {
        movshdup(xmm_aux3, xmm_dst); // dst:1,2,3,4; aux3:2,2,4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2),f(2,2),f(3,4),f(4,4)
        movhlps(xmm_aux3, xmm_dst);  // aux3:f(3,4),f(4,4),4,4
        horiz_ps(xmm_dst, xmm_aux3); // dst:f(1,2,3,4),...
        switch (dst_dt) {
            case memory::f32:
                movss(xmm_aux3, ptr[reg_dst]);
                horiz_ps(xmm_dst, xmm_aux3);
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::s32:
                movss(xmm_aux3, ptr[reg_dst]);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                movss(ptr[reg_dst], xmm_dst);
                break;
            case memory::u8:
                vpbroadcastb(xmm_aux3, ptr[reg_dst]);
                uni_vpmovzxbd(xmm_aux3, xmm_aux3);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            case memory::s8:
                vpbroadcastb(xmm_aux3, ptr[reg_dst]);
                uni_vpmovsxbd(xmm_aux3, xmm_aux3);
                uni_vcvtdq2ps(xmm_aux3, xmm_aux3);
                horiz_ps(xmm_dst, xmm_aux3);
                uni_vcvtps2dq(xmm_dst, xmm_dst);
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                pextrb(ptr[reg_dst], xmm_dst, 0);
                break;
            default:
                assert(!"unknown dst_dt");
        }
    }

    inline void horiz_ps(const Xmm& xmm, const Operand& op) {
        switch (jcp_.reduce_mode) {
            case Reduce::And:
                andps(xmm, op);
                break;
            case Reduce::L1:
            case Reduce::L2:
            case Reduce::LogSum:
            case Reduce::Mean:
            case Reduce::Sum:
            case Reduce::SumSquare:
            case Reduce::LogSumExp:
                addps(xmm, op);
                break;
            case Reduce::Max:
                maxps(xmm, op);
                break;
            case Reduce::Min:
                minps(xmm, op);
                break;
            case Reduce::Or:
                orps(xmm, op);
                break;
            case Reduce::Prod:
                mulps(xmm, op);
                break;
            default:
                assert(!"unsupported reduce mode");
        }
    }
};

MKLDNNReduceNode::MKLDNNReduceNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNReduceNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != 2)
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << " gets incorrect number of input edges!";
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << " gets incorrect number of output edges!";

    if (getParentEdgeAt(REDUCE_INDEXES)->getDims().ndims() != 1) {
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    auto *layer = getCnnLayer().get();
    keep_dims = layer->GetParamAsBool("keep_dims", false);

    if (keep_dims) {
        if (getParentEdgeAt(REDUCE_DATA)->getDims().ndims() != getChildEdgeAt(0)->getDims().ndims())
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets incorrect number of input/output dimensions!";
    } else {
        if (getParentEdgeAt(REDUCE_DATA)->getDims().ndims() <= getChildEdgeAt(0)->getDims().ndims())
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets incorrect number of input/output dimensions!";
    }

    Type reduce_mode = getType();
    if (reduce_mode == ReduceAnd) reduceMode = Reduce::And;
    else if (reduce_mode == ReduceL1) reduceMode = Reduce::L1;
    else if (reduce_mode == ReduceL2) reduceMode = Reduce::L2;
    else if (reduce_mode == ReduceLogSum) reduceMode = Reduce::LogSum;
    else if (reduce_mode == ReduceLogSumExp) reduceMode = Reduce::LogSumExp;
    else if (reduce_mode == ReduceMax) reduceMode = Reduce::Max;
    else if (reduce_mode == ReduceMean) reduceMode = Reduce::Mean;
    else if (reduce_mode == ReduceMin) reduceMode = Reduce::Min;
    else if (reduce_mode == ReduceOr) reduceMode = Reduce::Or;
    else if (reduce_mode == ReduceProd) reduceMode = Reduce::Prod;
    else if (reduce_mode == ReduceSum) reduceMode = Reduce::Sum;
    else if (reduce_mode == ReduceSumSquare) reduceMode = Reduce::SumSquare;
    else
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << " gets unsupported Reduce layer type!";
}

void MKLDNNReduceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrecision = getCnnLayer()->insData[REDUCE_DATA].lock()->getPrecision();
    Precision outputPrecision = getCnnLayer()->outData[0]->getPrecision();

    if (inputPrecision == Precision::BF16) inputPrecision = Precision::FP32;
    if (outputPrecision == Precision::BF16) outputPrecision = Precision::FP32;

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    input_prec = inputPrecision;
    output_prec = outputPrecision;
    src_data_size = MKLDNNExtensionUtils::sizeOfDataType(inputDataType);
    dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(outputDataType);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[REDUCE_DATA].constant = false;
    config.inConfs[REDUCE_INDEXES].constant = false;
    config.outConfs[0].constant = false;
    config.inConfs[REDUCE_DATA].inPlace = -1;
    config.inConfs[REDUCE_INDEXES].inPlace = -1;
    config.outConfs[0].inPlace = -1;

    auto pushDesc = [&](memory::format inFormat, memory::format outFormat, memory::data_type inDataType, memory::data_type outDataType) {
        config.inConfs[REDUCE_DATA].desc = MKLDNNMemoryDesc(getParentEdgeAt(REDUCE_DATA)->getDims(), inDataType, inFormat);
        config.inConfs[REDUCE_INDEXES].desc = MKLDNNMemoryDesc(getParentEdgeAt(REDUCE_INDEXES)->getDims(), memory::s32, memory::x);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outDataType, outFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, outFormat});
    };

    jit_mode = (mayiuse(cpu::sse42)) && getParentEdgeAt(REDUCE_DATA)->getDims().ndims() <= 5 &&
            (inputPrecision == Precision::FP32 || inputPrecision == Precision::I32 || inputPrecision == Precision::U8 || inputPrecision == Precision::I8) &&
            (outputPrecision == Precision::FP32 || outputPrecision == Precision::I32 || outputPrecision == Precision::U8 || outputPrecision == Precision::I8);
    if (jit_mode) {
        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(REDUCE_DATA)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(0)->getDims().ndims())), inputDataType, outputDataType);
        if (keep_dims) {
            if (getParentEdgeAt(REDUCE_DATA)->getDims().ndims() == 4 && getParentEdgeAt(REDUCE_DATA)->getDims().ToSizeVector()[1] > 1) {
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nChw16c, memory::nChw16c, inputDataType, outputDataType);
                } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                    pushDesc(memory::nChw8c, memory::nChw8c, inputDataType, outputDataType);
                }
            } else if (getParentEdgeAt(REDUCE_DATA)->getDims().ndims() == 5 && getParentEdgeAt(REDUCE_DATA)->getDims().ToSizeVector()[1] > 1) {
                if (mayiuse(cpu::avx512_common)) {
                    pushDesc(memory::nCdhw16c, memory::nCdhw16c, inputDataType, outputDataType);
                } else if (mayiuse(cpu::avx2) || mayiuse(cpu::sse42)) {
                    pushDesc(memory::nCdhw8c, memory::nCdhw8c, inputDataType, outputDataType);
                }
            }
        }
    } else {
        pushDesc(MKLDNNMemory::GetPlainFormat(memory::dims(getParentEdgeAt(REDUCE_DATA)->getDims().ndims())),
             MKLDNNMemory::GetPlainFormat(memory::dims(getChildEdgeAt(0)->getDims().ndims())), memory::f32, memory::f32);
    }
}

void MKLDNNReduceNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcDataMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();
    auto &srcIndexesMemPtr = getParentEdgeAt(REDUCE_INDEXES)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "didn't allocate destination memory.";
    if (!srcDataMemPtr || !srcDataMemPtr->GetPrimitivePtr() || !srcIndexesMemPtr || !srcIndexesMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "didn't allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "didn't set preferable primitive descriptor.";

    auto selectedPD = getSelectedPrimitiveDescriptor();
    Layout selected_layout = selectedPD->getConfig().inConfs[REDUCE_DATA].desc.getLayout();
    planar_layout = MKLDNNMemory::GetPlainLayout(getParentEdgeAt(REDUCE_DATA)->getDims()) == selected_layout;

    auto jcp = jit_reduce_config_params();
    jcp.src_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().inConfs[REDUCE_DATA].desc.getPrecision());
    jcp.dst_dt = MKLDNNExtensionUtils::IEPrecisionToDataType(selectedPD->getConfig().outConfs[0].desc.getPrecision());
    jcp.src_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.src_dt);
    jcp.dst_data_size = MKLDNNExtensionUtils::sizeOfDataType(jcp.dst_dt);
    jcp.planar_layout = planar_layout;
    jcp.reduce_mode = reduceMode;

    if (mayiuse(cpu::avx512_common)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::avx512_common>(jcp));
        reduce_post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::avx512_common>(jcp));
        blk_size = 16;
    } else if (mayiuse(cpu::avx2)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::avx2>(jcp));
        reduce_post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::avx2>(jcp));
        blk_size = 8;
    } else if (mayiuse(cpu::sse42)) {
        reduce_kernel.reset(new jit_uni_reduce_kernel_f32<cpu::sse42>(jcp));
        reduce_post_kernel.reset(new jit_uni_reduce_post_kernel_f32<cpu::sse42>(jcp));
        blk_size = 8;
    }

    jit_mode = jit_mode && reduce_kernel;
}

void MKLDNNReduceNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();
    auto &srcIndexesMemPtr = getParentEdgeAt(REDUCE_INDEXES)->getMemoryPtr();

    const auto idx_data = reinterpret_cast<const int32_t *>(srcIndexesMemPtr->GetData());
    size_t dst_size = dstMemPtr->GetSize();
    src_dims = getParentEdgeAt(REDUCE_DATA)->getDesc().getDims();
    src_strides = getParentEdgeAt(REDUCE_DATA)->getDesc().getBlockingDesc().getStrides();
    dims_size = src_dims.size();
    calc_process_dst_dims(idx_data);

    if (dims_size <= 5) {
        if (dims_size == 5) {
            SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], src_dims[2], src_dims[3], src_dims[4]);
            SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], process_dst_dims[2], process_dst_dims[3], process_dst_dims[4]);
        } else if (dims_size == 4) {
            SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], 1, src_dims[2], src_dims[3]);
            SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], 1, process_dst_dims[2], process_dst_dims[3]);
        } else if (dims_size == 3) {
            SET_SRC_DIM_VALUE(1, src_dims[0], 1, src_dims[1], src_dims[2]);
            SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, process_dst_dims[1], process_dst_dims[2]);
        } else if (dims_size == 2) {
            SET_SRC_DIM_VALUE(1, 1, 1, src_dims[0], src_dims[1]);
            SET_DST_DIM_VALUE(1, 1, 1, process_dst_dims[0], process_dst_dims[1]);
        } else {
            SET_SRC_DIM_VALUE(1, src_dims[0], 1, 1, 1);
            SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, 1, 1);
        }

        ReduceN = IB != OB && OB == 1;
        ReduceC = IC != OC && OC == 1;
        ReduceD = ID != OD && OD == 1;
        ReduceH = IH != OH && OH == 1;
        ReduceW = IW != OW && OW == 1;
    }

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->GetData()) +
                   srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
                   MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(srcMemPtr->GetDescriptor().data.data_type));
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->GetData()) +
                   dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding *
                   MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dstMemPtr->GetDescriptor().data.data_type));
    if (jit_mode) {
        reduce_type(src_data, dst_data, dst_size);
    } else {
        if (planar_layout) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            reduce_ref(in_ptr, out_ptr);
        } else {
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "only supports plain layout on machine w/o sse42.";
        }
    }
}

void MKLDNNReduceNode::reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size) {
    init_dst_data(out_ptr, dst_size);

    if (planar_layout) {
        reduce_PLN(in_ptr, out_ptr);
    } else {
        if ((reduceMode == Reduce::And || reduceMode == Reduce::LogSumExp || reduceMode == Reduce::Max ||
             reduceMode == Reduce::Min || reduceMode == Reduce::Prod) && ReduceC) {
            reduce_BLK_concern_padding(in_ptr, out_ptr);
        } else {
            reduce_BLK(in_ptr, out_ptr);
        }
    }
}

void MKLDNNReduceNode::reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr) {
    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_PLN;
        if (!ReduceC && !ReduceD && ReduceH && ReduceW) {
            parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW, 1);
            });
        } else if (ReduceH && ReduceW) {
            for (size_t ic = 0; ic < IC; ic++) {
                size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                    reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW, 1);
                }
            }
        } else if (!ReduceH && ReduceW) {
            for (size_t ic = 0; ic < IC; ic++) {
                size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                    parallel_for(IH, [&](size_t ih){
                        size_t oh = ih; GET_PTR_NCDH_PLN;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                    });
                }
            }
        } else if (ReduceW) {
            for (size_t ic = 0; ic < IC; ic++) {
                size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                    }
                }
            }
        } else {
            for (size_t ic = 0; ic < IC; ic++) {
                size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                        for (size_t ibw = 0; ibw < IW / blk_size; ibw++) {
                            size_t obw = ibw;
                            reduce_kernel_process(in_ptr_ncdh + ibw * blk_size * src_data_size,
                                                  out_ptr_ncdh + obw * blk_size * dst_data_size, blk_size, 0);
                        }
                        size_t tail_start = IW / blk_size * blk_size;
                        reduce_kernel_process(in_ptr_ncdh + tail_start * src_data_size, out_ptr_ncdh + tail_start * dst_data_size, IW - tail_start, 0);
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

void MKLDNNReduceNode::reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceC && !ReduceD && ReduceH && ReduceW) {
            parallel_for2d(ICB, ID, [&](size_t icb, size_t id) {
                size_t ocb = icb, od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
            });
        } else if (ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
                }
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                    }
                }
            }
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        parallel_for(IW, [&](size_t iw) {
                            size_t ow = iw; GET_PTR_NCDHW_BLK;
                            reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                        });
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

void MKLDNNReduceNode::reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);

    auto reduceSkipPadding = [&](const uint8_t *in_ptr_ncd, uint8_t *out_ptr_ncd, size_t ic) {
        size_t blk_valid_size = IC - ic;
        for (size_t ih = 0; ih < IH; ih++) {
            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
            for (size_t iw = 0; iw < IW; iw++) {
                size_t ow = ReduceW ? 0 : iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_valid_size);
            }
        }
    };

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                parallel_for(ID, [&](size_t id) {
                    size_t od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                    if (ic + blk_size <= IC) {
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                });
            }
        } else if (ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                if (ic + blk_size <= IC) {
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                } else {
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = 0; GET_PTR_NCD_BLK;
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            parallel_for(IW, [&](size_t iw) {
                                size_t ow = iw; GET_PTR_NCDHW_BLK;
                                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                            });
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        }
    }

    reduce_kernel_post_process(out_ptr);
}

inline void MKLDNNReduceNode::reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount, size_t reduce_w) {
    auto arg = jit_reduce_call_args();
    arg.src = static_cast<const void *>(in_p);
    arg.dst = static_cast<void *>(out_p);
    arg.work_amount = work_amount;
    arg.reduce_w = reduce_w;
    (*reduce_kernel)(&arg);
}

inline void MKLDNNReduceNode::reduce_kernel_post_process(uint8_t *out_ptr) {
    const float divisor = static_cast<float>(IB * IC * ID * IH * IW / (OB * OC * OD * OH * OW));
    if (planar_layout) {
        size_t parallel_amount = OB * OC * OD;
        parallel_for(parallel_amount, [&](size_t i) {
            uint8_t *out_p = out_ptr + i * OH * OW * dst_data_size;
            auto arg = jit_reduce_call_args();
            arg.dst = static_cast<void *>(out_p);
            arg.reduce_c = 2;
            arg.work_amount = OH * OW;
            arg.divisor = &divisor;
            (*reduce_post_kernel)(&arg);
        });
    } else {
        size_t OCB = div_up(OC, blk_size);
        size_t parallel_amount = OB * OCB * OD;
        parallel_for(parallel_amount, [&](size_t i) {
            uint8_t *out_p = out_ptr + i * OH * OW * blk_size * dst_data_size;
            auto arg = jit_reduce_call_args();
            arg.dst = static_cast<void *>(out_p);
            arg.reduce_c = ReduceC ? 1 : 0;
            arg.work_amount = OH * OW * blk_size;
            arg.divisor = &divisor;
            (*reduce_post_kernel)(&arg);
        });
    }
}

inline void MKLDNNReduceNode::init_dst_data(uint8_t *out_ptr, size_t dst_size) {
    switch (reduceMode) {
        case Reduce::L1:
        case Reduce::L2:
        case Reduce::LogSum:
        case Reduce::LogSumExp:
        case Reduce::Mean:
        case Reduce::Or:
        case Reduce::Sum:
        case Reduce::SumSquare:
            memset(out_ptr, 0, dst_size);
            break;
        case Reduce::And:
        case Reduce::Prod:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<float>(1); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int32_t>(1); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<uint8_t>(1); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int8_t>(1); });
            }
            break;
        case Reduce::Max:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::min(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::min(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::min(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::min(); });
            }
            break;
        case Reduce::Min:
            if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::max(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::max(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::max(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::max(); });
            }
            break;
        default:
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets unsupported reduce mode.";
    }
}

inline void MKLDNNReduceNode::calc_process_dst_dims(const int32_t *idx_data) {
    SizeVector out_dims;
    SizeVector dst_dims = getChildEdgeAt(0)->getDesc().getDims();
    std::set<size_t> axes;
    for (size_t i = 0; i < getParentEdgeAt(REDUCE_INDEXES)->getDims()[0]; i++) {
        int32_t axis = idx_data[i];
        if (axis < 0)
            axis += src_dims.size();
        if (static_cast<size_t>(axis) > src_dims.size())
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < src_dims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (keep_dims) out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(src_dims[i]);
            process_dst_dims.push_back(src_dims[i]);
        }
    }
    for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
        if (out_dims[i] != dst_dims[i])
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets incorrect number of output dimensions!";
    }
}

inline void MKLDNNReduceNode::reduce_ref(const float *in_ptr, float *out_ptr) {
    switch (reduceMode) {
        case Reduce::And:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x && y; });
            break;
        case Reduce::L1:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + (y >= 0 ? y : -y); });
            break;
        case Reduce::L2:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
        case Reduce::LogSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Reduce::LogSumExp:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + expf(y); });
            break;
        case Reduce::Max:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::min(),
                                                    [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Reduce::Mean:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Reduce::Min:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::max(),
                                                    [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Reduce::Or:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x || y; });
            break;
        case Reduce::Prod:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x * y; });
            break;
        case Reduce::Sum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Reduce::SumSquare:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
    default:
        THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets unsupported reduce mode.";
    }
}

void MKLDNNReduceNode::reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func) {
    size_t work_amount_dst = 1, reduced_dims_work_amount = 1;
    for (size_t i = 0; i < process_dst_dims.size(); i++)
        work_amount_dst *= process_dst_dims[i];
    for (size_t i = 0; i < src_dims.size(); i++)
        reduced_dims_work_amount *= src_dims[i];
    reduced_dims_work_amount /= work_amount_dst;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        SizeVector dst_counters(process_dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = process_dst_dims.size() - 1, i = start; j >= 0; j--) {
            dst_counters[j] = i % process_dst_dims[j];
            i /= process_dst_dims[j];
        }
        for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
            float reduce_prod = init_value;
            bool update_idx = true;
            SizeVector src_counters = dst_counters;
            for (i = 0; i < reduced_dims_work_amount; ++i) {
                if (update_idx) {
                    src_idx = 0;
                    for (j = 0; j < static_cast<int>(src_dims.size()); ++j)
                        src_idx += (src_counters[j] % src_dims[j]) * src_strides[j];
                    update_idx = false;
                }
                reduce_prod = func(reduce_prod, in_ptr[src_idx]);
                for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                    src_counters[axes_for_reduction[j]]++;
                    if (src_counters[axes_for_reduction[j]] < src_dims[axes_for_reduction[j]]) {
                        src_idx += src_strides[axes_for_reduction[j]];
                        break;
                    } else {
                        src_counters[axes_for_reduction[j]] = 0;
                        update_idx = true;
                    }
                }
            }
            out_ptr[dst_idx] = reduce_prod;
            for (j = process_dst_dims.size() - 1; j >= 0; j--) {
                dst_counters[j]++;
                if (dst_counters[j] < process_dst_dims[j])
                    break;
                else
                    dst_counters[j] = 0;
            }
        }
    });

    reduce_ref_map(out_ptr, work_amount_dst, reduced_dims_work_amount);
}

inline void MKLDNNReduceNode::reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount) {
    switch (reduceMode) {
        case Reduce::And:
        case Reduce::L1:
        case Reduce::Max:
        case Reduce::Min:
        case Reduce::Or:
        case Reduce::Prod:
        case Reduce::Sum:
        case Reduce::SumSquare:
            break;
        case Reduce::L2:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = std::sqrt(out_ptr[i]);
            });
            break;
        case Reduce::LogSum:
        case Reduce::LogSumExp:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = logf(out_ptr[i]);
            });
            break;
        case Reduce::Mean:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] /= reduced_dims_work_amount;
            });
            break;
        default:
            THROW_IE_EXCEPTION << "Reduce layer with name " << getName() << "gets unsupported reduce mode.";
    }
}

bool MKLDNNReduceNode::created() const {
    return getType() == ReduceAnd || getType() == ReduceL1 || getType() == ReduceL2 ||
           getType() == ReduceLogSum || getType() == ReduceLogSumExp || getType() == ReduceMax ||
           getType() == ReduceMean || getType() == ReduceMin || getType() == ReduceOr ||
           getType() == ReduceProd || getType() == ReduceSum || getType() == ReduceSumSquare;
}

REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceAnd);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceL1);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceL2);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceLogSum);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceLogSumExp);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceMax);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceMean);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceMin);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceOr);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceProd);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceSum);
REG_MKLDNN_PRIM_FOR(MKLDNNReduceNode, ReduceSumSquare);
