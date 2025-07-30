// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize.h"

#include <memory_desc/cpu_memory_desc_utils.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <common/c_types_map.hpp>
#include <common/dnnl_thread.hpp>
#include <common/nstl.hpp>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer_legacy.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <xbyak/xbyak.h>

#    include "cpu/x64/jit_generator.hpp"
#endif

// Quantization ranges validation is switched off by default in order to avoid regressions on user side
// #define VALIDATE_QUANTIZATION_RANGES

// Uncomment it to compute scales and shifts in double precision
// #define FQ_DOUBLE_PRECISION

using namespace dnnl;
using namespace ov;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu::node {
#if defined(OPENVINO_ARCH_X86_64)
#    define GET_OFF(field) offsetof(jit_quantize_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_binarization_kernel : public jit_uni_quantize_kernel, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binarization_kernel)

    explicit jit_uni_binarization_kernel(const jit_quantize_params& jqp)
        : jit_uni_quantize_kernel(jqp),
          jit_generator_t(jit_name()) {}

    void create_ker() override {
        jit_generator_t::create_kernel();
        ker_ = jit_kernel_cast<decltype(ker_)>(jit_ker());
    };

    void generate() override {
        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_thresholds, ptr[param + GET_OFF(thresholds)]);
        mov(reg_output_mask, ptr[param + GET_OFF(output_mask)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        const int nbits = 8;
        int simd_w = isa == avx512_core ? 16 : 8;
        const int C = jqp_.c;
        const int tail_size = C % simd_w;

        Label unrolled_loop_label;
        Label main_loop_label;
        Label tail_label;
        Label exit_label;

        L(unrolled_loop_label);
        {
            auto [step, ur_ch] = [&]() {
                if (isa == cpu::x64::sse41) {
                    return std::make_pair(nbits / 2, nbits);
                }
                if (isa == cpu::x64::avx2) {
                    return std::make_pair(nbits, nbits / 2);
                }
                return std::make_pair(2 * nbits, nbits / 4);
            }();
            const int unrolled_loop_step = ur_ch * step;

            cmp(reg_work_amount, unrolled_loop_step);
            jl(main_loop_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int ch = 0; ch < ur_ch; ch++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + ch * step * sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + ch * step * sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + ch * step * sizeof(float)]);
                if (isa == avx512_core) {
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

            add(reg_from, unrolled_loop_step * sizeof(float));
            add(reg_thresholds, unrolled_loop_step * sizeof(float));
            add(reg_output_mask, unrolled_loop_step * sizeof(float));
            add(reg_to, sizeof(uint32_t));
            sub(reg_work_amount, unrolled_loop_step);

            jmp(unrolled_loop_label, T_NEAR);
        }

        L(main_loop_label);
        {
            auto [repeats, step] = [&]() {
                if (isa == cpu::x64::sse41) {
                    return std::make_pair(2, nbits / 2);
                }
                if (isa == cpu::x64::avx2) {
                    return std::make_pair(1, nbits);
                }
                return std::make_pair(1, nbits * 2);
            }();
            const int main_loop_step = step * repeats;

            cmp(reg_work_amount, main_loop_step);
            jl(tail_label, T_NEAR);

            xor_(reg_bin_32, reg_bin_32);
            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src(0), ptr[reg_from + i * step * sizeof(float)]);
                uni_vmovups(vmm_wei(0), ptr[reg_thresholds + i * step * sizeof(float)]);
                uni_vmovups(vmm_mask(0), ptr[reg_output_mask + i * step * sizeof(float)]);
                if (isa == avx512_core) {
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
            if (isa == avx512_core) {
                mov(ptr[reg_to], reg_bin_16);
            } else {
                mov(ptr[reg_to], reg_bin_8);
            }

            add(reg_from, main_loop_step * sizeof(float));
            add(reg_thresholds, main_loop_step * sizeof(float));
            add(reg_output_mask, main_loop_step * sizeof(float));
            add(reg_to, isa == avx512_core ? sizeof(uint16_t) : sizeof(uint8_t));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_label);
        {
            if (tail_size != 0) {
                xor_(reg_bin_32, reg_bin_32);
                mov(reg_mask, 1);
                for (int c = 0; c < tail_size; c++) {
                    uni_vpxor(xmm_src(0), xmm_src(0), xmm_src(0));
                    uni_vpxor(xmm_wei(0), xmm_wei(0), xmm_wei(0));
                    uni_vpxor(xmm_mask(0), xmm_mask(0), xmm_mask(0));

                    uni_vmovss(xmm_src(0), ptr[reg_from + c * sizeof(float)]);
                    uni_vmovss(xmm_wei(0), ptr[reg_thresholds + c * sizeof(float)]);
                    uni_vmovss(xmm_mask(0), ptr[reg_output_mask + c * sizeof(float)]);
                    uni_vcmpgtps(xmm_src(0), xmm_src(0), xmm_wei(0));
                    uni_vpcmpeqd(xmm_src(0), xmm_src(0), xmm_mask(0));
                    uni_vmovmskps(reg_src_32, xmm_src(0));

                    shl(reg_src_32, c);
                    and_(reg_src_32, reg_mask);
                    or_(reg_bin_32, reg_src_32);
                    shl(reg_mask, 1);
                }
                if (isa == avx512_core && tail_size > nbits) {
                    mov(ptr[reg_to], reg_bin_16);
                } else {
                    mov(ptr[reg_to], reg_bin_8);
                }
            }
        }

        L(exit_label);

        this->postamble();
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    Vmm vmm_src(int idx) {
        return Vmm(idx);
    }
    Xmm xmm_src(int idx) {
        return Xmm(idx);
    }
    Vmm vmm_wei(int idx) {
        return Vmm(idx + 4);
    }
    Vmm vmm_mask(int idx) {
        return Vmm(idx + 5);
    }
    Xmm xmm_wei(int idx) {
        return Xmm(idx + 4);
    }
    Xmm xmm_mask(int idx) {
        return Xmm(idx + 5);
    }

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
struct jit_uni_quantization_kernel : public jit_uni_quantize_kernel, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_quantization_kernel)

    explicit jit_uni_quantization_kernel(const jit_quantize_params& jqp)
        : jit_uni_quantize_kernel(jqp),
          jit_generator_t(jit_name()) {}

    void create_ker() override {
        jit_generator_t::create_kernel();
        ker_ = jit_kernel_cast<decltype(ker_)>(jit_ker());
    };

    void generate() override {
        do_dequantization = jqp_.op_type == Algorithm::FQCommon;
        do_rounding = do_dequantization || jqp_.dst_prc == ov::element::f32;

        this->preamble();

        if (jqp_.is_planar) {
            compute_planar();
        } else {
            compute_generic();
        }

        this->postamble();
    }

private:
    using Vmm =
        typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    Vmm vmm_val(int idx) {
        return Vmm(idx + 0);
    }
    Vmm vmm_crop_low(int idx) {
        return Vmm(idx + 2);
    }
    Vmm vmm_crop_high(int idx) {
        return Vmm(idx + 4);
    }
    Vmm vmm_input_scale(int idx) {
        return Vmm(idx + 6);
    }
    Vmm vmm_input_shift(int idx) {
        return Vmm(idx + 8);
    }
    Vmm vmm_output_scale(int idx) {
        return Vmm(idx + 10);
    }
    Vmm vmm_output_shift(int idx) {
        return Vmm(idx + 12);
    }

    Ymm ymm_val(int idx) {
        return Ymm(idx + 0);
    }
    Ymm ymm_crop_low(int idx) {
        return Ymm(idx + 2);
    }
    Ymm ymm_crop_high(int idx) {
        return Ymm(idx + 4);
    }
    Ymm ymm_input_scale(int idx) {
        return Ymm(idx + 6);
    }
    Ymm ymm_input_shift(int idx) {
        return Ymm(idx + 8);
    }
    Ymm ymm_output_scale(int idx) {
        return Ymm(idx + 10);
    }
    Ymm ymm_output_shift(int idx) {
        return Ymm(idx + 12);
    }

    Xmm xmm_val(int idx) {
        return Xmm(idx + 0);
    }
    Xmm xmm_crop_low(int idx) {
        return Xmm(idx + 2);
    }
    Xmm xmm_crop_high(int idx) {
        return Xmm(idx + 4);
    }
    Xmm xmm_input_scale(int idx) {
        return Xmm(idx + 6);
    }
    Xmm xmm_input_shift(int idx) {
        return Xmm(idx + 8);
    }
    Xmm xmm_output_scale(int idx) {
        return Xmm(idx + 10);
    }
    Xmm xmm_output_shift(int idx) {
        return Xmm(idx + 12);
    }

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

    bool do_rounding = true;
    bool do_dequantization = true;

    void load_broadcasted_vectors_only(size_t idx) {
        const auto& broadcasted = jqp_.broadcasted;
        if (broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)]) {
            uni_vbroadcastss(vmm_crop_low(idx), ptr[reg_crop_low]);
        }
        if (broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)]) {
            uni_vbroadcastss(vmm_crop_high(idx), ptr[reg_crop_high]);
        }
        if (broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)]) {
            uni_vbroadcastss(vmm_input_scale(idx), ptr[reg_input_scale]);
        }
        if (broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)]) {
            uni_vbroadcastss(vmm_input_shift(idx), ptr[reg_input_shift]);
        }
        if (do_dequantization) {
            if (broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)]) {
                uni_vbroadcastss(vmm_output_scale(idx), ptr[reg_output_scale]);
            }
            if (broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)]) {
                uni_vbroadcastss(vmm_output_shift(idx), ptr[reg_output_shift]);
            }
        }
    }

    template <typename T>
    void load_not_broadcasted_vectors_only(size_t idx, size_t offset) {
        const auto& broadcasted = jqp_.broadcasted;
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)]) {
            uni_vmovups(T(vmm_crop_low(idx).getIdx()), ptr[reg_crop_low + offset]);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)]) {
            uni_vmovups(T(vmm_crop_high(idx).getIdx()), ptr[reg_crop_high + offset]);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)]) {
            uni_vmovups(T(vmm_input_scale(idx).getIdx()), ptr[reg_input_scale + offset]);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)]) {
            uni_vmovups(T(vmm_input_shift(idx).getIdx()), ptr[reg_input_shift + offset]);
        }
        if (do_dequantization) {
            if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)]) {
                uni_vmovups(T(vmm_output_scale(idx).getIdx()), ptr[reg_output_scale + offset]);
            }
            if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)]) {
                uni_vmovups(T(vmm_output_shift(idx).getIdx()), ptr[reg_output_shift + offset]);
            }
        }
    }

    void increase_ptrs_if_not_broadcasted(size_t offset) {
        const auto& broadcasted = jqp_.broadcasted;
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)]) {
            add(reg_crop_low, offset);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)]) {
            add(reg_crop_high, offset);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)]) {
            add(reg_input_scale, offset);
        }
        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)]) {
            add(reg_input_shift, offset);
        }
        if (do_dequantization) {
            if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)]) {
                add(reg_output_scale, offset);
            }
            if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)]) {
                add(reg_output_shift, offset);
            }
        }
    }

    void compute_planar() {
        int src_type_size = jqp_.src_prc.size();
        int dst_type_size = jqp_.dst_prc.size();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);

        mov(reg_crop_low, ptr[param + GET_OFF(crop_low)]);
        mov(reg_crop_high, ptr[param + GET_OFF(crop_high)]);
        mov(reg_input_scale, ptr[param + GET_OFF(input_scale)]);
        mov(reg_input_shift, ptr[param + GET_OFF(input_shift)]);
        mov(reg_output_scale, ptr[param + GET_OFF(output_scale)]);
        mov(reg_output_shift, ptr[param + GET_OFF(output_shift)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        if (isa == cpu::x64::avx512_core) {
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        }

        int simd_w = isa == cpu::x64::avx512_core ? 16 : 8;
        int tail_simd_w = 4;
        int repeats = isa == cpu::x64::sse41 ? 2 : 1;

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

        L(main_loop_label);
        {
            cmp(reg_work_amount, simd_w);
            jl(tail_blk4_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], jqp_.src_prc);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(0));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(0));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(0), vmm_input_shift(0));
                if (do_rounding) {
                    uni_vroundps(vmm_val(i), vmm_val(i), 0);
                }
                if (do_dequantization) {
                    uni_vfmadd213ps(vmm_val(i), vmm_output_scale(0), vmm_output_shift(0));
                }

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), jqp_.dst_prc);
            }

            sub(reg_work_amount, simd_w);
            add(reg_from, simd_w * src_type_size);
            add(reg_to, simd_w * dst_type_size);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_blk4_label);
        {
            cmp(reg_work_amount, tail_simd_w);
            jl(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) {
                uni_vroundps(xmm_val(0), xmm_val(0), 0);
            }
            if (do_dequantization) {
                uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));
            }

            store_vector(ptr[reg_to], xmm_val(0), jqp_.dst_prc);

            sub(reg_work_amount, tail_simd_w);
            add(reg_from, tail_simd_w * src_type_size);
            add(reg_to, tail_simd_w * dst_type_size);
        }

        L(tail_blk4_exit_label);

        mov(aux_reg_from, reg_from);
        mov(aux_reg_to, reg_to);

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            load_scalar(xmm_val(0), ptr[aux_reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) {
                uni_vroundps(xmm_val(0), xmm_val(0), 0);
            }
            if (do_dequantization) {
                uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));
            }

            store_scalar(ptr[aux_reg_to], xmm_val(0), jqp_.dst_prc);

            sub(reg_work_amount, 1);
            add(aux_reg_from, 1 * src_type_size);
            add(aux_reg_to, 1 * dst_type_size);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    void compute_generic() {
        int src_type_size = jqp_.src_prc.size();
        int wei_type_size = jqp_.wei_prc.size();
        int dst_type_size = jqp_.dst_prc.size();

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

        if (isa == cpu::x64::avx512_core) {
            uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        }

        constexpr unsigned simd_w = isa == cpu::x64::avx512_core ? 16 : 8;
        constexpr unsigned tail8_simd_w = 8;
        constexpr unsigned tail4_simd_w = 4;
        constexpr int repeats = isa == cpu::x64::sse41 ? 2 : 1;

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

        for (int i = 0; i < repeats; i++) {
            load_broadcasted_vectors_only(i);
        }

        cmp(reg_block_size, simd_w);
        jl(simd_w == 16 ? tail_blk8_label : tail_blk4_label, T_NEAR);

        for (int i = 0; i < repeats; i++) {
            load_not_broadcasted_vectors_only<Vmm>(i, i * (simd_w / 2) * sizeof(float));
        }

        L(main_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                load_vector(vmm_val(i), ptr[reg_from + i * (simd_w / 2) * src_type_size], jqp_.src_prc);

                uni_vminps(vmm_val(i), vmm_val(i), vmm_crop_high(i));
                uni_vmaxps(vmm_val(i), vmm_val(i), vmm_crop_low(i));
                uni_vfmadd213ps(vmm_val(i), vmm_input_scale(i), vmm_input_shift(i));
                if (do_rounding) {
                    uni_vroundps(vmm_val(i), vmm_val(i), 0);
                }
                if (do_dequantization) {
                    uni_vfmadd213ps(vmm_val(i), vmm_output_scale(i), vmm_output_shift(i));
                }

                store_vector(ptr[reg_to + i * (simd_w / 2) * dst_type_size], vmm_val(i), jqp_.dst_prc);
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

            load_not_broadcasted_vectors_only<Ymm>(0, 0);

            L(tail_blk8_loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(tail_blk8_exit_label, T_NEAR);

                load_vector(ymm_val(0), ptr[aux_reg_from], jqp_.src_prc);

                uni_vminps(ymm_val(0), ymm_val(0), ymm_crop_high(0));
                uni_vmaxps(ymm_val(0), ymm_val(0), ymm_crop_low(0));
                uni_vfmadd213ps(ymm_val(0), ymm_input_scale(0), ymm_input_shift(0));
                if (do_rounding) {
                    uni_vroundps(ymm_val(0), ymm_val(0), 0);
                }
                if (do_dequantization) {
                    uni_vfmadd213ps(ymm_val(0), ymm_output_scale(0), ymm_output_shift(0));
                }

                store_vector(ptr[aux_reg_to], ymm_val(0), jqp_.dst_prc);

                dec(reg_work_amount);
                add(aux_reg_from, reg_src_step);
                add(aux_reg_to, reg_dst_step);

                jmp(tail_blk8_loop_label, T_NEAR);
            }

            L(tail_blk8_exit_label);

            add(reg_from, tail8_simd_w * src_type_size);
            add(reg_to, tail8_simd_w * dst_type_size);
            increase_ptrs_if_not_broadcasted(tail8_simd_w * wei_type_size);
            sub(reg_block_size, tail8_simd_w);
        }

        L(tail_blk4_label);

        cmp(reg_block_size, tail4_simd_w);
        jl(tail_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        load_not_broadcasted_vectors_only<Xmm>(0, 0);

        L(tail_blk4_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(tail_blk4_exit_label, T_NEAR);

            load_vector(xmm_val(0), ptr[aux_reg_from], jqp_.src_prc);

            uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
            uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
            uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
            if (do_rounding) {
                uni_vroundps(xmm_val(0), xmm_val(0), 0);
            }
            if (do_dequantization) {
                uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));
            }

            store_vector(ptr[aux_reg_to], xmm_val(0), jqp_.dst_prc);

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_blk4_loop_label, T_NEAR);
        }

        L(tail_blk4_exit_label);

        add(reg_from, tail4_simd_w * src_type_size);
        add(reg_to, tail4_simd_w * dst_type_size);
        increase_ptrs_if_not_broadcasted(tail4_simd_w * wei_type_size);
        sub(reg_block_size, tail4_simd_w);

        L(tail_label);

        cmp(reg_block_size, 0);
        jle(exit_label, T_NEAR);

        mov(aux_reg_to, reg_to);
        mov(aux_reg_from, reg_from);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        L(tail_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);
            Label end_unroll;

            auto tail_unroll = [&](size_t iter) {
                const auto& broadcasted = jqp_.broadcasted;
                for (size_t i = 0; i < iter; i++) {
                    if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)]) {
                        uni_vmovss(xmm_crop_low(0), ptr[reg_crop_low + i * wei_type_size]);
                    }
                    if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)]) {
                        uni_vmovss(xmm_crop_high(0), ptr[reg_crop_high + i * wei_type_size]);
                    }
                    if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)]) {
                        uni_vmovss(xmm_input_scale(0), ptr[reg_input_scale + i * wei_type_size]);
                    }
                    if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)]) {
                        uni_vmovss(xmm_input_shift(0), ptr[reg_input_shift + i * wei_type_size]);
                    }
                    if (do_dequantization) {
                        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)]) {
                            uni_vmovss(xmm_output_scale(0), ptr[reg_output_scale + i * wei_type_size]);
                        }
                        if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)]) {
                            uni_vmovss(xmm_output_shift(0), ptr[reg_output_shift + i * wei_type_size]);
                        }
                    }

                    load_scalar(xmm_val(0), ptr[aux_reg_from + i * src_type_size], jqp_.src_prc);

                    uni_vminps(xmm_val(0), xmm_val(0), xmm_crop_high(0));
                    uni_vmaxps(xmm_val(0), xmm_val(0), xmm_crop_low(0));
                    uni_vfmadd213ps(xmm_val(0), xmm_input_scale(0), xmm_input_shift(0));
                    if (do_rounding) {
                        uni_vroundps(xmm_val(0), xmm_val(0), 0);
                    }
                    if (do_dequantization) {
                        uni_vfmadd213ps(xmm_val(0), xmm_output_scale(0), xmm_output_shift(0));
                    }

                    store_scalar(ptr[aux_reg_to + i * dst_type_size], xmm_val(0), jqp_.dst_prc);
                }
                jmp(end_unroll, T_NEAR);
            };

            std::array<Label, tail4_simd_w> unroll_labels;
            for (size_t i = 1; i < tail4_simd_w; ++i) {
                cmp(reg_block_size, i);
                je(unroll_labels[i], T_NEAR);
            }

            for (size_t i = 1; i < tail4_simd_w; ++i) {
                L(unroll_labels[i]);
                tail_unroll(i);
            }

            L(end_unroll);

            dec(reg_work_amount);
            add(aux_reg_from, reg_src_step);
            add(aux_reg_to, reg_dst_step);

            jmp(tail_loop_label, T_NEAR);
        }

        L(exit_label);
    }

    void load_vector(Zmm zmm_src, const Xbyak::Address& op, ov::element::Type src_prc) {
        switch (src_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(zmm_src, op);
            break;
        case ov::element::i8:
            uni_vpmovsxbd(zmm_src, op);
            break;
        case ov::element::u8:
            uni_vpmovzxbd(zmm_src, op);
            break;
        default:
            assert(!"unknown src_prc");
        }

        if (src_prc != ov::element::f32) {
            uni_vcvtdq2ps(zmm_src, zmm_src);
        }
    }

    void load_vector(Ymm ymm_src, const Xbyak::Address& op, ov::element::Type src_prc) {
        switch (src_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(ymm_src, op);
            break;
        case ov::element::i8:
            uni_vpmovsxbd(ymm_src, op);
            break;
        case ov::element::u8:
            uni_vpmovzxbd(ymm_src, op);
            break;
        default:
            assert(!"unknown src_prc");
        }

        if (src_prc != ov::element::f32) {
            uni_vcvtdq2ps(ymm_src, ymm_src);
        }
    }

    void load_vector(Xmm xmm_src, const Xbyak::Address& op, ov::element::Type src_prc) {
        switch (src_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(xmm_src, op);
            break;
        case ov::element::i8:
            uni_vpmovsxbd(xmm_src, op);
            break;
        case ov::element::u8:
            uni_vpmovzxbd(xmm_src, op);
            break;
        default:
            assert(!"unknown src_prc");
        }

        if (src_prc != ov::element::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    void load_scalar(Xmm xmm_src, const Xbyak::Address& op, ov::element::Type src_prc) {
        switch (src_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovss(xmm_src, op);
            break;
        case ov::element::i8:
            movsx(reg_tmp_32, op);
            uni_vmovq(xmm_src, reg_tmp_64);
            break;
        case ov::element::u8:
            movzx(reg_tmp_32, op);
            uni_vmovq(xmm_src, reg_tmp_64);
            break;
        default:
            assert(!"unknown src_prc");
        }

        if (src_prc != ov::element::f32) {
            uni_vcvtdq2ps(xmm_src, xmm_src);
        }
    }

    void store_vector(const Xbyak::Address& op, Zmm zmm_dst, ov::element::Type dst_prc) {
        if (dst_prc != ov::element::f32) {
            uni_vcvtps2dq(zmm_dst, zmm_dst);
        }

        switch (dst_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(op, zmm_dst);
            break;
        case ov::element::i8:
            vpmovsdb(op, zmm_dst);
            break;
        case ov::element::u8:
            vpmaxsd(zmm_dst, zmm_dst, vmm_zero);
            vpmovusdb(op, zmm_dst);
            break;
        default:
            assert(!"unknown dst_prc");
        }
    }

    void store_vector(const Xbyak::Address& op, Ymm ymm_dst, ov::element::Type dst_prc) {
        auto xmm_dst = Xmm(ymm_dst.getIdx());

        if (dst_prc != ov::element::f32) {
            uni_vcvtps2dq(ymm_dst, ymm_dst);
        }

        switch (dst_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(op, ymm_dst);
            break;
        case ov::element::i8:
            uni_vpackssdw(ymm_dst, ymm_dst, ymm_dst);

            vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpacksswb(ymm_dst, ymm_dst, ymm_dst);

            vmovq(op, xmm_dst);
            break;
        case ov::element::u8:
            uni_vpackusdw(ymm_dst, ymm_dst, ymm_dst);

            vpermq(ymm_dst, ymm_dst, 0x08);

            uni_vpackuswb(ymm_dst, ymm_dst, ymm_dst);

            vmovq(op, xmm_dst);
            break;
        default:
            assert(!"unknown dst_prc");
        }
    }

    void store_vector(const Xbyak::Address& op, Xmm xmm_dst, ov::element::Type dst_prc) {
        if (dst_prc != ov::element::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovups(op, xmm_dst);
            break;
        case ov::element::i8:
            uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
            uni_vmovd(op, xmm_dst);
            break;
        case ov::element::u8:
            uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
            uni_vmovd(op, xmm_dst);
            break;
        default:
            assert(!"unknown dst_prc");
        }
    }

    void store_scalar(const Xbyak::Address& op, Xmm xmm_dst, ov::element::Type dst_prc) {
        if (dst_prc != ov::element::f32) {
            uni_vcvtps2dq(xmm_dst, xmm_dst);
        }

        switch (dst_prc) {
        case ov::element::f32:
        case ov::element::i32:
            uni_vmovss(op, xmm_dst);
            break;
        case ov::element::i8:
            uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
            uni_vmovq(reg_tmp_64, xmm_dst);
            mov(op, reg_tmp_8);
            break;
        case ov::element::u8:
            uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
            uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
            uni_vmovq(reg_tmp_64, xmm_dst);
            mov(op, reg_tmp_8);
            break;
        default:
            assert(!"unknown dst_prc");
        }
    }
};
#endif
bool FakeQuantize::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto fq = ov::as_type_ptr<const ov::op::v0::FakeQuantize>(op);
        if (!fq) {
            errorMessage = "Only v0 FakeQuantize operation is supported";
            return false;
        }
        const auto dataRank = fq->get_input_partial_shape(0).rank().get_length();
        if (dataRank < 2 || dataRank > 5) {
            errorMessage = "Doesn't support 'data' input with rank: " + std::to_string(dataRank);
            return false;
        }
        for (size_t i = 1; i < fq->get_input_size(); i++) {
            if (fq->get_input_partial_shape(i).rank().get_length() > 5) {
                errorMessage = "Doesn't support 'range' input with rank: " +
                               std::to_string(fq->get_input_partial_shape(i).rank().get_length());
                return false;
            }
        }
        for (size_t i = 1; i < fq->get_input_size(); i++) {
            if (!ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(i))) {
                errorMessage = "Has non const 'range' input on " + std::to_string(i) + " port";
                return false;
            }
        }
        for (size_t i = 1; i < fq->get_input_size(); i++) {
            size_t count_not_unit_axis = 0;
            auto shape = getNormalizedDimsBySize(fq->get_input_shape(i), dataRank);

            if (ov::shape_size(shape) != 1) {
                size_t not_unit_axis = 0;
                for (size_t i = 0; i < shape.size(); i++) {
                    if (shape[i] > 1) {
                        not_unit_axis = i;
                        count_not_unit_axis++;
                    }
                }

                /* @todo
                 * Channel axis 2 is added for 3D MatMul (most common one).
                 * FQ for non-1 channel fallbacks to reference implementation.
                 * Expected to be fused for 3D MatMul
                 * Long term idea: restore limitation for channel axis 1 and
                 * support fusing of unfolded FQ (see FakeQuantizeDecomposition transformation)
                 */
                if (count_not_unit_axis > 1 || none_of(not_unit_axis, 1U, 2U)) {
                    errorMessage = "Supports only per-tensor and per-channel quantizations";
                    return false;
                }
            }
        }
        if (none_of(fq->get_auto_broadcast().m_type,
                    ov::op::AutoBroadcastType::NONE,
                    ov::op::AutoBroadcastType::NUMPY)) {
            errorMessage = "Doesn't support broadcast type: " + ov::as_string(fq->get_auto_broadcast().m_type);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
struct FakeQuantKey {
    jit_quantize_params jqp;
    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        // using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        seed = hash_combine(seed, jqp.is_planar);
        seed = hash_combine(seed, jqp.src_prc.hash());
        seed = hash_combine(seed, jqp.wei_prc.hash());
        seed = hash_combine(seed, jqp.dst_prc.hash());
        seed = hash_combine(seed, jqp.op_type);
        if (jqp.op_type == Algorithm::FQBinarization) {
            seed = hash_combine(seed, jqp.c);
        } else {
            seed = hash_combine(seed, jqp.broadcasted);
        }
        return seed;
    }

    bool operator==(const FakeQuantKey& rhs) const {
        bool result = jqp.is_planar == rhs.jqp.is_planar && jqp.src_prc == rhs.jqp.src_prc &&
                      jqp.wei_prc == rhs.jqp.wei_prc && jqp.dst_prc == rhs.jqp.dst_prc &&
                      jqp.op_type == rhs.jqp.op_type;
        if (result) {
            if (jqp.op_type == Algorithm::FQBinarization) {
                result = result && jqp.c == rhs.jqp.c;
            } else {
                result = result && jqp.broadcasted == rhs.jqp.broadcasted;
            }
        }
        return result;
    }
};
}  // namespace

FakeQuantize::FakeQuantize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        algorithm = Algorithm::FQCommon;
        const auto fq = ov::as_type_ptr<const ov::op::v0::FakeQuantize>(op);

        levels = fq->get_levels();
        CPU_NODE_ASSERT(levels > 1, "supports 'levels' attribute greater than or equal to 2");

        CPU_NODE_ASSERT(inputShapes.size() == 5, "has incorrect number of input edges: ", inputShapes.size());
        CPU_NODE_ASSERT(outputShapes.size() == 1, "has incorrect number of output edges: ", outputShapes.size());

        auto initAxisIdx = [&](const VectorDims& inputDims) {
            size_t axisIdx = 0;
            for (size_t i = 1; i < inputDims.size(); i++) {
                if (inputDims[i] > 1) {
                    axisIdx = i;
                }
            }

            return axisIdx;
        };

        const size_t dataRank = getInputShapeAtPort(0).getRank();
        axis = dataRank == 1 ? 0 : 1;
        int axisSize = -1;

        const auto ilShape = getNormalizedDimsBySize(fq->get_input_shape(1), dataRank);
        auto inputLowAxis = initAxisIdx(ilShape);
        isInputLowBroadcasted = (ov::is_scalar(ilShape) || ilShape[inputLowAxis] == 1);
        if (!isInputLowBroadcasted) {
            axis = inputLowAxis;
            axisSize = ilShape[inputLowAxis];
        }

        const auto ihShape = getNormalizedDimsBySize(fq->get_input_shape(2), dataRank);
        auto inputHighAxis = initAxisIdx(ihShape);
        isInputHighBroadcasted = (ov::is_scalar(ihShape) || ihShape[inputHighAxis] == 1);
        if (!isInputHighBroadcasted) {
            axis = inputHighAxis;
            axisSize = ihShape[inputHighAxis];
        }

        const auto olShape = getNormalizedDimsBySize(fq->get_input_shape(3), dataRank);
        auto outputLowAxis = initAxisIdx(olShape);
        isOutputLowBroadcasted = (ov::is_scalar(olShape) || olShape[outputLowAxis] == 1);
        if (!isOutputLowBroadcasted) {
            axis = outputLowAxis;
            axisSize = olShape[outputLowAxis];
        }

        const auto ohShape = getNormalizedDimsBySize(fq->get_input_shape(4), dataRank);
        auto outputHighAxis = initAxisIdx(ohShape);
        isOutputHighBroadcasted = (ov::is_scalar(ohShape) || ohShape[outputHighAxis] == 1);
        if (!isOutputHighBroadcasted) {
            axis = outputHighAxis;
            axisSize = ohShape[outputHighAxis];
        }

        auto inputLowAxisSize = ov::is_scalar(ilShape) ? 1 : ilShape[inputLowAxis];
        auto inputHighAxisSize = ov::is_scalar(ihShape) ? 1 : ihShape[inputHighAxis];
        auto outputLowAxisSize = ov::is_scalar(olShape) ? 1 : olShape[outputLowAxis];
        auto outputHighAxisSize = ov::is_scalar(ohShape) ? 1 : ohShape[outputHighAxis];

        CPU_NODE_ASSERT(axisSize == -1 || dimsEqualWeak(axisSize, getInputShapeAtPort(0).getDims()[axis]),
                        "has different quantization axis size on 'data' and 'range' inputs");

        const auto inputLowNode = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(1));
        auto inputLowData = inputLowNode->cast_vector<float>();

        const auto inputHighNode = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(2));
        auto inputHighData = inputHighNode->cast_vector<float>();

        const auto outputLowNode = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(3));
        auto outputLowData = outputLowNode->cast_vector<float>();

        const auto outputHighNode = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(4));
        auto outputHighData = outputHighNode->cast_vector<float>();

        binarization = levels == 2;

        if (binarization) {
            for (size_t i = 0; i < outputLowAxisSize; i++) {
                if (none_of(outputLowData[i], 1.F, 0.F)) {
                    binarization = false;
                    break;
                }
            }

            for (size_t i = 0; i < outputHighAxisSize; i++) {
                if (none_of(outputHighData[i], 1.F, 0.F)) {
                    binarization = false;
                    break;
                }
            }

            for (size_t i = 0; i < std::max(inputLowAxisSize, inputHighAxisSize); i++) {
                if (inputLowData[isInputLowBroadcasted ? 0 : i] != inputHighData[isInputHighBroadcasted ? 0 : i]) {
                    binarization = false;
                    break;
                }
            }
        }

        if (binarization) {
            algorithm = Algorithm::FQBinarization;

            if (isInputLowBroadcasted) {
                binarizationThresholds.push_back(inputLowData[0]);
            } else {
                CPU_NODE_ASSERT(axisSize != -1, "axisSize is not set");
                binarizationThresholds.resize(rnd_up(axisSize, 16));
                for (int i = 0; i < axisSize; i++) {
                    binarizationThresholds[i] = inputLowData[i];
                }
            }

            if (isOutputHighBroadcasted) {
                binarizationOutputMask.push_back(outputHighData[0] == 1.F ? 0xffffffff : 0x00000000);
            } else {
                CPU_NODE_ASSERT(axisSize != -1, "axisSize is not set");
                binarizationOutputMask.resize(rnd_up(axisSize, 16));
                for (int i = 0; i < axisSize; i++) {
                    binarizationOutputMask[i] = outputHighData[i] == 1.F ? 0xffffffff : 0x00000000;
                }
            }
        } else {
            auto allElementsAreEqual = [&](const std::vector<float>& data, size_t size) {
                if (size == 0) {
                    return true;
                }

                auto first = data[0];
                for (size_t i = 1; i < size; i++) {
                    if (data[i] != first) {
                        return false;
                    }
                }

                return true;
            };

            if (allElementsAreEqual(inputLowData, inputLowAxisSize)) {
                inputLowAxisSize = 1;
                isInputLowBroadcasted = true;
            }

            if (allElementsAreEqual(inputHighData, inputHighAxisSize)) {
                inputHighAxisSize = 1;
                isInputHighBroadcasted = true;
            }

            if (allElementsAreEqual(outputLowData, outputLowAxisSize)) {
                outputLowAxisSize = 1;
                isOutputLowBroadcasted = true;
            }

            if (allElementsAreEqual(outputHighData, outputHighAxisSize)) {
                outputHighAxisSize = 1;
                isOutputHighBroadcasted = true;
            }

            cropLow.resize(inputLowAxisSize);
            cropHigh.resize(inputHighAxisSize);
            inputScale.resize(std::max(inputLowAxisSize, inputHighAxisSize));
            inputShift.resize(std::max(inputLowAxisSize, inputHighAxisSize));
            outputScale.resize(std::max(outputLowAxisSize, outputHighAxisSize));
            outputShift.resize(outputLowAxisSize);

            cropLowSize = cropLow.size();
            cropHighSize = cropHigh.size();
            inputScaleSize = inputScale.size();
            inputShiftSize = inputShift.size();
            outputScaleSize = outputScale.size();
            outputShiftSize = outputShift.size();

            broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)] = cropLowSize == 1;
            broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)] = cropHighSize == 1;
            broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)] = inputScaleSize == 1;
            broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)] = inputShiftSize == 1;
            broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)] = outputScaleSize == 1;
            broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)] = outputShiftSize == 1;

            if (all_of(1U,
                       cropLowSize,
                       cropHighSize,
                       inputScaleSize,
                       inputShiftSize,
                       outputScaleSize,
                       outputShiftSize)) {
                broadcastingPolicy = PerTensor;
            } else if (any_of(1U,
                              cropLowSize,
                              cropHighSize,
                              inputScaleSize,
                              inputShiftSize,
                              outputScaleSize,
                              outputShiftSize)) {
                broadcastingPolicy = Mixed;
            } else {
                broadcastingPolicy = PerChannel;
            }

            bool quantizationOnly = true;

            for (size_t i = 0; i < cropLow.size(); i++) {
                cropLow[i] = inputLowData[isInputLowBroadcasted ? 0 : i];
            }

            for (size_t i = 0; i < cropHigh.size(); i++) {
                cropHigh[i] = inputHighData[isInputHighBroadcasted ? 0 : i];
            }

            for (size_t i = 0; i < inputScale.size(); i++) {
                float il = inputLowData[isInputLowBroadcasted ? 0 : i];
                float ih = inputHighData[isInputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
                if ((il == ih && levels != 2) || il > ih || std::isnan(il) || std::isnan(ih) || std::isinf(il) ||
                    std::isinf(ih)) {
                    CPU_NODE_THROW("has invalid input quantize ranges: ", "inputLow = ", il, ", inputHigh = ", ih);
                }
#endif
#ifdef FQ_DOUBLE_PRECISION
                inputScale[i] = (levels - 1.0) / (static_cast<double>(ih) - il);
                inputShift[i] = -il * (levels - 1.0) / (static_cast<double>(ih) - il);
#else
                inputScale[i] = (levels - 1) / (ih - il);
                inputShift[i] = -il * (levels - 1) / (ih - il);
#endif
            }

            for (size_t i = 0; i < outputScale.size(); i++) {
                float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
                float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

#if defined(VALIDATE_QUANTIZATION_RANGES)
                if (std::isnan(ol) || std::isnan(oh) || std::isinf(ol) || std::isinf(oh)) {
                    CPU_NODE_THROW("has wrong output quantize ranges: ", "outputLow = ", ol, ", outputHigh = ", oh);
                }
#endif
#ifdef FQ_DOUBLE_PRECISION
                outputScale[i] = (static_cast<double>(oh) - ol) / (levels - 1.0);
#else
                outputScale[i] = (oh - ol) / (levels - 1);
#endif

                if (outputScale[i] != 1.F) {
                    quantizationOnly = false;
                }
            }

            for (size_t i = 0; i < outputShift.size(); i++) {
                float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];

                outputShift[i] = ol;

                if (outputShift[i] != 0.F) {
                    quantizationOnly = false;
                }
            }

            bool isFakeQuantization = true;
            bool isFakeQuantizationWithScale = true;
            for (size_t i = 0;
                 i < std::max(inputLowAxisSize,
                              std::max(outputLowAxisSize, std::max(inputHighAxisSize, outputHighAxisSize)));
                 i++) {
                float il = inputLowData[isInputLowBroadcasted ? 0 : i];
                float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
                float ih = inputHighData[isInputHighBroadcasted ? 0 : i];
                float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

                isFakeQuantization = isFakeQuantization && il == ol && ih == oh;
                isFakeQuantizationWithScale = isFakeQuantizationWithScale && il != ih && ol != oh &&
                                              (abs(ol / (oh - ol) - il / (ih - il)) < 0.001F);
            }

            if (isFakeQuantizationWithScale) {
                for (size_t i = 0;
                     i < std::max(inputLowAxisSize,
                                  std::max(outputLowAxisSize, std::max(inputHighAxisSize, outputHighAxisSize)));
                     i++) {
                    float il = inputLowData[isInputLowBroadcasted ? 0 : i];
                    float ol = outputLowData[isOutputLowBroadcasted ? 0 : i];
                    float ih = inputHighData[isInputHighBroadcasted ? 0 : i];
                    float oh = outputHighData[isOutputHighBroadcasted ? 0 : i];

                    fqScales.push_back(1 / ((ih - il) / (oh - ol)));
                }
            }

            algorithm = quantizationOnly ? Algorithm::FQQuantization : Algorithm::FQCommon;
        }
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

std::vector<LayoutType> FakeQuantize::getDataFormats() const {
    // Special case for first FQ in the network
    const auto& dims = getInputShapeAtPort(0).getDims();
    if (dims[getAxis()] == 3) {
        return {LayoutType::ncsp};
    }
    if (isBinarization()) {
        return {LayoutType::nspc};
    }
    if (any_of(dims.size(), 4U, 5U)) {
        if (getAxis() == 1) {
            auto blkFormat = mayiuse(cpu::x64::avx512_core) ? LayoutType::nCsp16c : LayoutType::nCsp8c;
            return {blkFormat, LayoutType::nspc, LayoutType::ncsp};
        }
        return {LayoutType::ncsp};
    }
    return {LayoutType::ncsp};
}

void FakeQuantize::init() {
    if (binarization) {
        inputPrecision = ov::element::f32;
        outputPrecision = ov::element::u1;
    } else {
        inputPrecision = getOriginalInputPrecisionAtPort(0);
        outputPrecision = getOriginalOutputPrecisionAtPort(0);

        if (none_of(inputPrecision, ov::element::f32, ov::element::u8, ov::element::i8)) {
            inputPrecision = ov::element::f32;
        }

        if (none_of(outputPrecision, ov::element::f32, ov::element::u8, ov::element::i8)) {
            outputPrecision = ov::element::f32;
        }
    }
}

void FakeQuantize::getSupportedDescriptors() {
    CPU_NODE_ASSERT(getParentEdges().size() == 5, "has incorrect number of input edges: ", getParentEdges().size());
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges: ", getChildEdges().size());

    CPU_NODE_ASSERT(getInputShapeAtPort(0).getRank() == getOutputShapeAtPort(0).getRank(),
                    "has different ranks for input and output tensors");

    if (isBinarization()) {
        CPU_NODE_ASSERT(getInputShapeAtPort(0).getRank() == 4UL, "doesn't support input/output rank != 4");
    }

    if (getAxis() != 1) {
        CPU_NODE_ASSERT(!isBinarization(), "doesn't support non per-tensor binarization for axis: ", getAxis());
        CPU_NODE_ASSERT(getAxis() == 0, "doesn't support non per-tensor quantization for axis: ", getAxis());
    }
}

void FakeQuantize::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    impl_desc_type impl_type = []() {
        if (mayiuse(cpu::x64::avx512_core)) {
            return impl_desc_type::jit_avx512;
        }
        if (mayiuse(cpu::x64::avx2)) {
            return impl_desc_type::jit_avx2;
        }
        if (mayiuse(cpu::x64::sse41)) {
            return impl_desc_type::jit_sse42;
        }
        return impl_desc_type::ref;
    }();
    if (!mayiuse(cpu::x64::sse41) || getAxis() != 1) {
        impl_type = impl_desc_type::ref;

        if (!isBinarization()) {
            inputPrecision = ov::element::f32;
            outputPrecision = ov::element::f32;
        }
    }

    std::vector<LayoutType> dataFormats;
    // reference implementation supports only planar format
    if (impl_type == impl_desc_type::ref) {
        dataFormats.push_back(LayoutType::ncsp);
    } else {
        dataFormats = getDataFormats();
    }

    for (auto& fmt : dataFormats) {
        NodeConfig config;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            PortConfig dataConfig;
            dataConfig.inPlace(-1);
            dataConfig.constant(false);

            if (i == 0) {
                auto descCreator = BlockedDescCreator::getCommonCreators().at(fmt);
                dataConfig.setMemDesc(descCreator->createSharedDesc(getInputPrecision(), getInputShapeAtPort(i)));
            } else {
                auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
                dataConfig.setMemDesc(descCreator->createSharedDesc(ov::element::f32, getInputShapeAtPort(i)));
            }
            config.inConfs.push_back(dataConfig);
        }

        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        auto descCreator = BlockedDescCreator::getCommonCreators().at(fmt);
        dataConfig.setMemDesc(descCreator->createSharedDesc(getOutputPrecision(), getOutputShapeAtPort(0)));
        config.outConfs.push_back(dataConfig);

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    }
}

bool FakeQuantize::needPrepareParams() const {
    if (isBinarization()) {
        const auto* selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
        CPU_NODE_ASSERT(selectedPrimitiveDescriptor, "doesn't have primitive descriptors.");

        if (internalBlobMemory.empty() ||
            (selectedPrimitiveDescriptor->getImplementationType() != impl_desc_type::ref && inputShapesModified())) {
            return true;
        }

        const auto axisSize = getParentEdgeAt(0)->getMemory().getStaticDims()[getAxis()];
        const auto newPaddedSize = rnd_up(axisSize, 16);
        const auto currPaddedSize = rnd_up(currentAxisSize, 16);

        return newPaddedSize != currPaddedSize ||
               ((isInputLowBroadcasted || isOutputHighBroadcasted) && axisSize != currentAxisSize);
    }
    return false;
}

void FakeQuantize::prepareParams() {
    if (isBinarization()) {
        const size_t axisSize = getParentEdgeAt(0)->getMemory().getShape().getStaticDims()[getAxis()];
        const size_t newPaddedSize = rnd_up(axisSize, 16);
        CPU_NODE_ASSERT(newPaddedSize != 0, "newPaddedSize is 0");

        if (internalBlobMemory.empty() || newPaddedSize != rnd_up(currentAxisSize, 16) ||
            ((isInputLowBroadcasted || isOutputHighBroadcasted) && axisSize != currentAxisSize)) {
            DnnlBlockedMemoryDesc weightsDataDesc(Shape(VectorDims{newPaddedSize}),
                                                  memory::data_type::f32,
                                                  memory::format_tag::x);
            constexpr size_t numBinFqIntBlob = 2;
            bool needUpdThr = false;
            bool needUpdMask = false;
            if (isInputLowBroadcasted && axisSize != currentAxisSize) {
                binarizationThresholds.resize(newPaddedSize);
                std::fill(binarizationThresholds.begin() + 1,
                          binarizationThresholds.begin() + axisSize,
                          binarizationThresholds[0]);
                std::fill(binarizationThresholds.begin() + axisSize, binarizationThresholds.end(), 0.F);
                needUpdThr = true;
            }

            if (isOutputHighBroadcasted && axisSize != currentAxisSize) {
                binarizationOutputMask.resize(newPaddedSize);
                std::fill(binarizationOutputMask.begin() + 1,
                          binarizationOutputMask.begin() + axisSize,
                          binarizationOutputMask[0]);
                std::fill(binarizationOutputMask.begin() + axisSize, binarizationOutputMask.end(), 0);
                needUpdMask = true;
            }

            if (internalBlobMemory.empty() || needUpdThr) {
                auto binarizationThresholdsDataMem =
                    std::make_shared<Memory>(getEngine(), weightsDataDesc, getBinarizationTresholdsPtr());
                if (internalBlobMemory.empty()) {
                    internalBlobMemory.push_back(binarizationThresholdsDataMem);
                } else {
                    internalBlobMemory[0] = binarizationThresholdsDataMem;
                }
            }

            if (internalBlobMemory.size() == (numBinFqIntBlob - 1) || needUpdMask) {
                auto binarizationMaskDataMem =
                    std::make_shared<Memory>(getEngine(), weightsDataDesc, getBinarizationOutputMaskPtr());
                if (internalBlobMemory.size() == (numBinFqIntBlob - 1)) {
                    internalBlobMemory.push_back(binarizationMaskDataMem);
                } else {
                    internalBlobMemory[1] = binarizationMaskDataMem;
                }
            }
        }
        currentAxisSize = axisSize;
    }
}

void FakeQuantize::createPrimitive() {
    Node::createPrimitive();
    auto* selectedPrimitiveDescriptor = getSelectedPrimitiveDescriptor();
    CPU_NODE_ASSERT(selectedPrimitiveDescriptor, "doesn't have primitive descriptors.");
    if (selectedPrimitiveDescriptor->getImplementationType() != impl_desc_type::ref) {
        const auto& config = getSelectedPrimitiveDescriptor()->getConfig();

        // Form FakeQuanKey
        FakeQuantKey key = {};
        key.jqp.src_prc = config.inConfs[0].getMemDesc()->getPrecision();
        key.jqp.wei_prc = ov::element::f32;
        key.jqp.dst_prc = config.outConfs[0].getMemDesc()->getPrecision();

        const auto& srcMemory = getParentEdgeAt(0)->getMemory();
        const auto& srcDesc = srcMemory.getDesc();

        key.jqp.is_planar = srcDesc.hasLayoutType(LayoutType::ncsp) && any_of(srcDesc.getShape().getRank(), 3U, 4U, 5U);
        key.jqp.op_type = getAlgorithm();

        if (isBinarization()) {
            const auto& inDims = srcMemory.getStaticDims();
            key.jqp.c = inDims.size() > 1 ? inDims[1] : 1;
        } else {
            // in case of blocked layout we need to extend vectors to prevent read from unallocated memory
            size_t paddedSize = 1;
            if (srcDesc.hasLayoutType(LayoutType::nCsp16c)) {
                paddedSize = 16;
            } else if (srcDesc.hasLayoutType(LayoutType::nCsp8c)) {
                paddedSize = 8;
            }
            if (paddedSize != 1) {
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)]) {
                    cropLow.resize(rnd_up(cropLow.size(), paddedSize));
                }
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)]) {
                    cropHigh.resize(rnd_up(cropHigh.size(), paddedSize));
                }
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)]) {
                    inputScale.resize(rnd_up(inputScale.size(), paddedSize));
                }
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)]) {
                    inputShift.resize(rnd_up(inputShift.size(), paddedSize));
                }
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)]) {
                    outputScale.resize(rnd_up(outputScale.size(), paddedSize));
                }
                if (!broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)]) {
                    outputShift.resize(rnd_up(outputShift.size(), paddedSize));
                }
            }

            key.jqp.broadcasted = broadcasted;
        }

        auto cache = context->getParamsCache();
        auto buildExecutor = [](const FakeQuantKey& key) {
            return std::make_shared<FakeQuantizeJitExecutor>(key.jqp);
        };
        auto result = cache->getOrCreate(key, buildExecutor);
        execPtr = result.first;
    }
}

void FakeQuantize::executeReference() {
    auto srcMemory = getSrcMemoryAtPort(0);
    auto dstMemory = getDstMemoryAtPort(0);

    const auto* src = srcMemory->getDataAs<const float>();

    auto srcDims = srcMemory->getStaticDims();
    auto dstDims = dstMemory->getStaticDims();

    auto s_str = srcMemory->getDescWithType<BlockedMemoryDesc>()->getStrides();
    auto d_str = dstMemory->getDescWithType<BlockedMemoryDesc>()->getStrides();

    const int N = srcDims[0];
    const int C = srcDims.size() > 1 ? srcDims[1] : 1;
    const int D = srcDims.size() == 5 ? srcDims[2] : 1;
    int H = 1, W = 1;
    if (srcDims.size() == 3) {
        H = srcDims[2];
    } else if (srcDims.size() > 3) {
        H = srcDims[srcDims.size() - 2];
    }
    if (srcDims.size() > 3) {
        W = srcDims[srcDims.size() - 1];
    }

    if (isBinarization()) {
        size_t tmp = s_str[s_str.size() - 1];
        for (int i = s_str.size() - 1; i > 1; i--) {
            s_str[i] = s_str[i - 1];
        }
        s_str[1] = tmp;

        tmp = d_str[d_str.size() - 1];
        for (int i = d_str.size() - 1; i > 1; i--) {
            d_str[i] = d_str[i - 1];
        }
        d_str[1] = tmp;

        auto* dst = dstMemory->getDataAs<uint8_t>();

        const int nbits = 8;
        const int CB = impl::utils::div_up(C, nbits);

        const auto* thresholds = internalBlobMemory[0]->getDataAs<const float>();
        const auto* output_mask = internalBlobMemory[1]->getDataAs<const uint32_t>();

        parallel_nd(N, CB, D, H, W, [&](dim_t n, dim_t cb, dim_t d, dim_t h, dim_t w) {
            uint8_t bin_val = 0x00;
            for (int c = static_cast<int>(cb) * nbits, shift = 0;
                 c < std::min(static_cast<int>(C), (static_cast<int>(cb) + 1) * nbits);
                 c++, shift++) {
                size_t src_off = n * s_str[0] + c * s_str[1];
                if (srcDims.size() == 4) {
                    src_off += h * s_str[2] + w * s_str[3];
                } else if (srcDims.size() == 5) {
                    src_off += d * s_str[2] + h * s_str[3] + w * s_str[4];
                }

                float val = src[src_off];
                float thr = thresholds[c];
                uint32_t out_mask = output_mask[c];

                uint32_t res = (val > thr) ? 0xffffffff : 0x00000000;

                auto bit = static_cast<uint8_t>(res == out_mask);
                bin_val |= (bit << shift);
            }
            size_t dst_off = n * d_str[0] + (cb * nbits) * d_str[1];
            if (dstDims.size() == 4) {
                dst_off += h * d_str[2] + w * d_str[3];
            } else if (dstDims.size() == 5) {
                dst_off += d * d_str[2] + h * d_str[3] + w * d_str[4];
            }

            dst[dst_off / nbits] = bin_val;
        });
    } else {
        auto* dst = dstMemory->getDataAs<float>();

        parallel_nd(N, C, D, H, W, [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) {
            size_t src_off = n * s_str[0];
            if (srcDims.size() == 5) {
                src_off += d * s_str[2] + h * s_str[3] + w * s_str[4];
            } else if (srcDims.size() == 4) {
                src_off += h * s_str[2] + w * s_str[3];
            } else if (srcDims.size() == 3) {
                src_off += h * s_str[2];
            } else if (srcDims.size() == 2) {
                src_off += c * s_str[1];
            }

            float src_val = src[src_off];

            int wei_idx = getAxis() == 0 ? static_cast<int>(n) : static_cast<int>(c);
            float cl = broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)] ? cropLow[0] : cropLow[wei_idx];
            float ch = broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)] ? cropHigh[0] : cropHigh[wei_idx];
            float isc =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)] ? inputScale[0] : inputScale[wei_idx];
            float ish =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)] ? inputShift[0] : inputShift[wei_idx];
            float osc = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)] ? outputScale[0]
                                                                                          : outputScale[wei_idx];
            float osh = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)] ? outputShift[0]
                                                                                          : outputShift[wei_idx];

            float dst_val = nstl::min(ch, nstl::max(cl, src_val));
            dst_val = dst_val * isc + ish;
            dst_val = roundf(dst_val);
            dst_val = dst_val * osc + osh;

            size_t dst_off = n * d_str[0] + c * d_str[1];
            if (dstDims.size() == 5) {
                dst_off += d * d_str[2] + h * d_str[3] + w * d_str[4];
            } else if (dstDims.size() == 4) {
                dst_off += h * d_str[2] + w * d_str[3];
            } else if (dstDims.size() == 3) {
                dst_off += h * d_str[2];
            }

            dst[dst_off] = dst_val;
        });
    }
}
void FakeQuantize::executeBinarization(const std::unique_ptr<jit_uni_quantize_kernel>& pKernel) const {
#if defined(OPENVINO_ARCH_X86_64)
    auto srcMemory = getSrcMemoryAtPort(0);
    auto dstMemory = getDstMemoryAtPort(0);

    const auto* src = srcMemory->getDataAs<const uint8_t>();
    auto* dst = dstMemory->getDataAs<uint8_t>();

    const auto* thresholds = internalBlobMemory[0]->getDataAs<const float>();
    const auto* output_mask = internalBlobMemory[1]->getDataAs<const float>();

    auto src_dims = srcMemory->getStaticDims();

    auto srcMemDesc = srcMemory->getDescWithType<BlockedMemoryDesc>();
    std::vector<size_t> s_str = srcMemDesc->getStrides();
    size_t tmp = s_str[s_str.size() - 1];
    for (int i = s_str.size() - 1; i > 1; i--) {
        s_str[i] = s_str[i - 1];
    }
    s_str[1] = tmp;

    const int N = src_dims[0];
    const int C = src_dims[1];
    const int H = src_dims[2];
    const int W = src_dims[3];

    int nbits = 8;

    parallel_nd(N, H, W, [&](dim_t n, dim_t h, dim_t w) {
        auto arg = jit_quantize_call_args();

        arg.from = &src[(n * s_str[0] + h * s_str[2] + w * s_str[3]) * sizeof(float)];
        arg.to = &dst[(n * s_str[0] + h * s_str[2] + w * s_str[3]) / nbits];
        arg.thresholds = &thresholds[0];
        arg.output_mask = &output_mask[0];
        arg.work_amount = static_cast<size_t>(C);

        (*pKernel)(&arg);
    });
#endif
}

void FakeQuantize::executeQuantization(const std::unique_ptr<jit_uni_quantize_kernel>& pKernel) const {
#if defined(OPENVINO_ARCH_X86_64)
    auto srcMemory = getSrcMemoryAtPort(0);
    auto dstMemory = getDstMemoryAtPort(0);

    const auto* src = srcMemory->getDataAs<const uint8_t>();
    auto* dst = dstMemory->getDataAs<uint8_t>();

    const auto& srcDesc = srcMemory->getDesc();
    auto srcDims = srcDesc.getShape().getStaticDims();

    bool is_blk_format = !srcDesc.hasLayoutType(LayoutType::nspc) && any_of(srcDesc.getShape().getRank(), 4U, 5U);
    int blk_size = 1;
    if (!(srcDesc.hasLayoutType(LayoutType::ncsp) && any_of(srcDesc.getShape().getRank(), 3U, 4U, 5U)) &&
        mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (!(srcDesc.hasLayoutType(LayoutType::ncsp) && any_of(srcDesc.getShape().getRank(), 3U, 4U, 5U))) {
        blk_size = 8;
    }

    const auto& jqp = pKernel->jqp_;
    auto src_type_size = jqp.src_prc.size();
    auto dst_type_size = jqp.dst_prc.size();

    auto srcMemDesc = srcMemory->getDescWithType<BlockedMemoryDesc>();
    auto s_str = srcMemDesc->getStrides();

    if (is_blk_format) {
        s_str[1] /= blk_size;
    }

    if (srcDesc.hasLayoutType(LayoutType::nspc) && any_of(srcDesc.getShape().getRank(), 4U, 5U)) {
        size_t tmp = s_str[s_str.size() - 1];
        for (int i = s_str.size() - 1; i > 1; i--) {
            s_str[i] = s_str[i - 1];
        }
        s_str[1] = tmp;
    }

    const int N = srcDims[0];
    const int C = srcDims[1];
    const int CB = div_up(C, blk_size);
    int D = 1, H = 1, W = 1;
    if (srcDims.size() == 5) {
        D = srcDims[2];
    }
    if (srcDims.size() == 3) {
        H = srcDims[2];
    } else if (srcDims.size() > 3) {
        H = srcDims[srcDims.size() - 2];
    }
    if (srcDims.size() > 3) {
        W = srcDims[srcDims.size() - 1];
    }

    if (srcDesc.hasLayoutType(LayoutType::ncsp) && srcDesc.getShape().getRank() == 3) {
        parallel_nd(N, CB, D, [&](dim_t n, dim_t cb, [[maybe_unused]] dim_t d) {
            auto arg = jit_quantize_call_args();

            int c = static_cast<int>(cb) * blk_size;

            size_t data_off = n * s_str[0] + c * s_str[1];

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)] ? cropLow.data() : &cropLow[c];
            arg.crop_high =
                broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)] ? cropHigh.data() : &cropHigh[c];
            arg.input_scale =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)] ? inputScale.data() : &inputScale[c];
            arg.input_shift =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)] ? inputShift.data() : &inputShift[c];
            arg.output_scale = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)] ? outputScale.data()
                                                                                                 : &outputScale[c];
            arg.output_shift = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)] ? outputShift.data()
                                                                                                 : &outputShift[c];

            arg.src_step = static_cast<size_t>(blk_size) * src_type_size;
            arg.dst_step = static_cast<size_t>(blk_size) * dst_type_size;
            arg.block_size = static_cast<size_t>(blk_size);
            arg.work_amount = static_cast<size_t>(H);

            (*pKernel)(&arg);
        });
    } else if (jqp.is_planar && srcDims.size() > 2) {
        const int batch_size = 256;
        const int B = div_up(H * W, batch_size);
        parallel_nd(N, CB, D, B, [&](dim_t n, dim_t cb, dim_t d, dim_t b) {
            auto arg = jit_quantize_call_args();

            const int c = static_cast<int>(cb) * blk_size;
            const int h = static_cast<int>(b) * batch_size / W;
            const int w = static_cast<int>(b) * batch_size % W;

            const size_t data_off = any_of(srcDims.size(), 3U, 4U)
                                        ? n * s_str[0] + c * s_str[1] + h * s_str[2] + w
                                        : n * s_str[0] + c * s_str[1] + d * s_str[2] + h * s_str[3] + w;

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)] ? cropLow.data() : &cropLow[c];
            arg.crop_high =
                broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)] ? cropHigh.data() : &cropHigh[c];
            arg.input_scale =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)] ? inputScale.data() : &inputScale[c];
            arg.input_shift =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)] ? inputShift.data() : &inputShift[c];
            arg.output_scale = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)] ? outputScale.data()
                                                                                                 : &outputScale[c];
            arg.output_shift = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)] ? outputShift.data()
                                                                                                 : &outputShift[c];

            arg.src_step =
                is_blk_format ? static_cast<size_t>(blk_size) * src_type_size : static_cast<size_t>(C) * src_type_size;
            arg.dst_step =
                is_blk_format ? static_cast<size_t>(blk_size) * dst_type_size : static_cast<size_t>(C) * dst_type_size;
            arg.block_size = is_blk_format ? static_cast<size_t>(blk_size) : nstl::min(blk_size, C - c);
            arg.work_amount = static_cast<size_t>(std::min(static_cast<dim_t>(batch_size), H * W - b * batch_size));

            (*pKernel)(&arg);
        });
    } else {
        parallel_nd_legacy(N, CB, D, H, [&](dim_t n, dim_t cb, dim_t d, dim_t h) {
            auto arg = jit_quantize_call_args();

            int c = static_cast<int>(cb) * blk_size;

            size_t data_off = 0;
            if (srcDims.size() == 2) {
                data_off = n * s_str[0] + c * s_str[1];
            } else if (any_of(srcDims.size(), 3U, 4U)) {
                data_off = n * s_str[0] + c * s_str[1] + h * s_str[2];
            } else {
                data_off = n * s_str[0] + c * s_str[1] + d * s_str[2] + h * s_str[3];
            }

            arg.from = &src[data_off * src_type_size];
            arg.to = &dst[data_off * dst_type_size];
            arg.crop_low = broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_LOW)] ? cropLow.data() : &cropLow[c];
            arg.crop_high =
                broadcasted[static_cast<size_t>(FQ_add_input_type::CROP_HIGH)] ? cropHigh.data() : &cropHigh[c];
            arg.input_scale =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SCALE)] ? inputScale.data() : &inputScale[c];
            arg.input_shift =
                broadcasted[static_cast<size_t>(FQ_add_input_type::INPUT_SHIFT)] ? inputShift.data() : &inputShift[c];
            arg.output_scale = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SCALE)] ? outputScale.data()
                                                                                                 : &outputScale[c];
            arg.output_shift = broadcasted[static_cast<size_t>(FQ_add_input_type::OUTPUT_SHIFT)] ? outputShift.data()
                                                                                                 : &outputShift[c];

            arg.src_step =
                is_blk_format ? static_cast<size_t>(blk_size) * src_type_size : static_cast<size_t>(C) * src_type_size;
            arg.dst_step =
                is_blk_format ? static_cast<size_t>(blk_size) * dst_type_size : static_cast<size_t>(C) * dst_type_size;
            arg.block_size =
                (is_blk_format && srcDims.size() != 2) ? static_cast<size_t>(blk_size) : nstl::min(blk_size, C - c);
            arg.work_amount = static_cast<size_t>(W);

            (*pKernel)(&arg);
        });
    }
#endif
}

void FakeQuantize::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void FakeQuantize::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (getSelectedPrimitiveDescriptor()->getImplementationType() != impl_desc_type::ref) {
        execPtr->exec(*this);
    } else {
        executeReference();
    }
}

void FakeQuantize::initializePostOpData(const VectorDims& dims, const size_t bufferAlignment, bool doRounding) {
    if (postOpDataVersion == parameterVersion) {
        return;
    }

    if (getAlgorithm() == Algorithm::FQBinarization) {
        const auto realAxisSize = dims[dims.size() > 1 ? 1 : 0];
        const auto axisPaddedSize = rnd_up(realAxisSize, bufferAlignment);
        binarizationThresholds.resize(axisPaddedSize, 0);
        binarizationOutputMask.resize(axisPaddedSize, 0);

        if (isInputLowBroadcasted) {
            std::fill(binarizationThresholds.begin() + 1,
                      binarizationThresholds.begin() + realAxisSize,
                      binarizationThresholds[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }
        if (isOutputHighBroadcasted) {
            std::fill(binarizationOutputMask.begin() + 1,
                      binarizationOutputMask.begin() + realAxisSize,
                      binarizationOutputMask[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }
    } else {
        updateOptimizedFormula(doRounding);
    }

    postOpDataVersion = parameterVersion;
}

void FakeQuantize::initializePostOpDataLegacy(const VectorDims& dims, const size_t bufferAlignment) {
    if (legacyPostOpDataVersion == parameterVersion) {
        return;
    }

    if (getAlgorithm() == Algorithm::FQBinarization) {
        const auto realAxisSize = dims[dims.size() > 1 ? 1 : 0];
        const auto axisPaddedSize = rnd_up(realAxisSize, bufferAlignment);

        binarizationThresholds.resize(axisPaddedSize, 0);
        binarizationOutputMask.resize(axisPaddedSize, 0);

        if (isInputLowBroadcasted) {
            std::fill(binarizationThresholds.begin() + 1,
                      binarizationThresholds.begin() + realAxisSize,
                      binarizationThresholds[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }
        if (isOutputHighBroadcasted) {
            std::fill(binarizationOutputMask.begin() + 1,
                      binarizationOutputMask.begin() + realAxisSize,
                      binarizationOutputMask[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }

    } else {
        quantizationData.insert(quantizationData.end(), cropLow.begin(), cropLow.end());
        quantizationData.insert(quantizationData.end(), cropHigh.begin(), cropHigh.end());
        quantizationData.insert(quantizationData.end(), inputScale.begin(), inputScale.end());
        quantizationData.insert(quantizationData.end(), inputShift.begin(), inputShift.end());
        quantizationData.insert(quantizationData.end(), outputScale.begin(), outputScale.end());
        quantizationData.insert(quantizationData.end(), outputShift.begin(), outputShift.end());
        quantizationDataSize = quantizationData.size();

        int bufferPaddingSize = rnd_up(outputShift.size(), bufferAlignment) - outputShift.size();
        quantizationData.resize(quantizationDataSize + bufferPaddingSize, 0);
    }

    legacyPostOpDataVersion = parameterVersion;
}

void FakeQuantize::appendMemory(const size_t dataSize,
                                const void* data,
                                MemoryPtr& memPtr,
                                std::vector<MemoryPtr>& postOpsMem) {
    if (!memPtr) {
        DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, {dataSize});
        memPtr = std::make_shared<Memory>(getEngine(), memoryDesc, data);

        postOpsMem.push_back(memPtr);
    }
}

void FakeQuantize::appendMemory([[maybe_unused]] const size_t dataSize,
                                const void* data,
                                [[maybe_unused]] MemoryPtr& memPtr,
                                std::vector<const void*>& postOpsMem) {
    postOpsMem.push_back(data);
}

template <typename T>
void FakeQuantize::appendPostOpsImpl(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<T>& postOpsMem) {
    // try to map fakeQuantizeNode using output scale & eltwise first
    // if failed, fallback to append_quantization()

    // oneDNN quantization_injectors assumes that quantization data memory is always aligned on 16
    // by length of AVX512 vector register which is also enough for AVX2 and SSE42 implementations.
    // Otherwise it can lead to buffer over-read and performance penalties due to denormals.
    const size_t bufferAlignment = 16;

    initializePostOpDataLegacy(postOpDims, bufferAlignment);

    if (getAlgorithm() == Algorithm::FQBinarization) {
        ops.append_binarization(dnnl::algorithm::binarization_depthwise,
                                static_cast<const float*>(binarizationThresholds.data()),
                                reinterpret_cast<const float*>(binarizationOutputMask.data()));
    } else {
        dnnl::algorithm alg = getAlgorithm() == Algorithm::FQQuantization
                                  ? dnnl::algorithm::quantization_quantize
                                  : dnnl::algorithm::quantization_quantize_dequantize;

        std::array<bool, 6> per_channel = {cropLowSize > 1,
                                           cropHighSize > 1,
                                           inputScaleSize > 1,
                                           inputShiftSize > 1,
                                           outputScaleSize > 1,
                                           outputShiftSize > 1};

        std::array<bool, 6> all_default = {false};
        all_default[0] = std::all_of(cropLow.cbegin(), cropLow.cend(), [](float val) {
            return val == 0.F;
        });
        all_default[1] = std::all_of(cropHigh.cbegin(), cropHigh.cend(), [](float val) {
            return val == 0.F;
        });
        all_default[2] = std::all_of(inputScale.cbegin(), inputScale.cend(), [](float val) {
            return val == 1.F;
        });
        all_default[3] = std::all_of(inputShift.cbegin(), inputShift.cend(), [](float val) {
            return val == 0.F;
        });
        all_default[4] = std::all_of(outputScale.cbegin(), outputScale.cend(), [](float val) {
            return val == 1.F;
        });
        all_default[5] = std::all_of(outputShift.cbegin(), outputShift.cend(), [](float val) {
            return val == 0.F;
        });

        std::array<size_t, 6> offsets = {0};
        offsets[1] = offsets[0] + cropLowSize;
        offsets[2] = offsets[1] + cropHighSize;
        offsets[3] = offsets[2] + inputScaleSize;
        offsets[4] = offsets[3] + inputShiftSize;
        offsets[5] = offsets[4] + outputScaleSize;

        ops.append_quantization(alg, per_channel, all_default, offsets);

        appendMemory(quantizationDataSize, quantizationData.data(), quantizationMemory, postOpsMem);
    }
}

void FakeQuantize::appendPostOps(dnnl::post_ops& ops,
                                 const VectorDims& postOpDims,
                                 std::unordered_map<int, MemoryPtr>& postOpsMem,
                                 [[maybe_unused]] const int channelAxis) {
    std::vector<MemoryPtr> postOpsMemPtrs;
    appendPostOpsImpl(ops, postOpDims, postOpsMemPtrs);

    CPU_NODE_ASSERT(postOpsMemPtrs.size() <= 1, "at most 1 post ops memory args can be appended.");

    if (!postOpsMemPtrs.empty()) {
        postOpsMem[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = postOpsMemPtrs[0];
    }
}

void FakeQuantize::appendPostOps(dnnl::post_ops& ops,
                                 const VectorDims& postOpDims,
                                 std::vector<const void*>& postOpsMem,
                                 [[maybe_unused]] const int channelAxis) {
    appendPostOpsImpl(ops, postOpDims, postOpsMem);
}

static float roundHalfToEven(float f) {
    const float RHAFZ = std::round(f);  // r is round-half-away-from-zero
    const float d = RHAFZ - f;          // f + d -> RHAFZ
    if (none_of(d, 0.5F, -0.5F)) {
        return RHAFZ;
    }

    // already even +/-1.5 -> +/-2
    if (std::fmod(RHAFZ, 2.0F) == 0.0F) {
        return RHAFZ;
    }

    // +/-2.5 -> +/-3, but we need it to to +/-2
    // RHAFZ (f+d) goes the wrong way, should be (f-d)
    return f - d;
}

void FakeQuantize::updateOptimizedFormula(bool do_rounding) {
    auto& f = optimizedFormula;

    auto isPerTensor =
        [](const std::vector<float>& v, float ref, const float zero_thr = std::numeric_limits<float>::min()) {
            return std::all_of(v.cbegin(), v.cend(), [&](float val) {
                return abs(val - ref) < zero_thr;
            });
        };
    size_t OC = std::max({inputScale.size(),
                          inputShift.size(),
                          cropLow.size(),
                          cropHigh.size(),
                          outputScale.size(),
                          outputShift.size()});

    CPU_NODE_ASSERT(any_of(inputScale.size(), 1U, OC), "inputScale.size() == ", inputScale.size());
    CPU_NODE_ASSERT(any_of(inputShift.size(), 1U, OC), "inputShift.size() == ", inputShift.size());
    CPU_NODE_ASSERT(any_of(cropLow.size(), 1U, OC), "cropLow.size() == ", cropLow.size());
    CPU_NODE_ASSERT(any_of(cropHigh.size(), 1U, OC), "cropHigh.size() == ", cropHigh.size());
    CPU_NODE_ASSERT(any_of(outputScale.size(), 1U, OC), "outputScale.size() == ", outputScale.size());
    CPU_NODE_ASSERT(any_of(outputShift.size(), 1U, OC), "outputShift.size() == ", outputShift.size());

    // WA: a per-Tensor input shift may little drift away randomly
    //     from it's orginal value when FQ was fused with any
    //     preceding per-channel multiply and create a false
    //     per-channel input shift, this threshold was chosen carefully
    //     to recorver the per-Tensor nature w/o mistaking a real
    //     per-channel FQ.
    if (isPerTensor(inputShift, inputShift[0], 0.00005F)) {
        f.ish.resize(OC);
        for (auto& v : f.ish) {
            v = inputShift[0];
        }
    } else {
        f.ish = inputShift;
    }
    f.clo = cropLow;
    f.chi = cropHigh;
    f.isc = inputScale;
    f.osc = outputScale;
    f.osh = outputShift;

    if (f.clo.size() == 1) {
        f.clo.resize(OC, f.clo[0]);
    }
    if (f.chi.size() == 1) {
        f.chi.resize(OC, f.chi[0]);
    }
    if (f.isc.size() == 1) {
        f.isc.resize(OC, f.isc[0]);
    }
    if (f.ish.size() == 1) {
        f.ish.resize(OC, f.ish[0]);
    }

    for (size_t i = 0; i < OC; i++) {
        auto& clo = f.clo[i];
        auto& chi = f.chi[i];
        auto& isc = f.isc[i];
        auto& ish = f.ish[i];
        const auto& osc = f.osc[f.osc.size() == 1 ? 0 : i];
        const auto& osh = f.osh[f.osh.size() == 1 ? 0 : i];

        clo = roundHalfToEven(clo * isc + ish);
        chi = roundHalfToEven(chi * isc + ish);
        if (clo > chi) {
            std::swap(clo, chi);
        }

        if (!do_rounding) {
            // when no rounding is needed, outputScale/outputShift can be
            // merged with inputScale/inputShift with updated cropLow/cropHigh
            clo = clo * osc + osh;
            chi = chi * osc + osh;
            if (clo > chi) {
                std::swap(clo, chi);
            }

            //  crop(x*isc + ish, a, b)*osc + osh
            //  crop(x*isc*osc + ish*osc + osh, a', b')
            isc = isc * osc;
            ish = ish * osc + osh;
        }
    }

    if (!do_rounding) {
        f.osc.clear();
        f.osh.clear();
    }

    f.shrinkLength();

    if (f.osc.size() == 1 && f.osc[0] == 1.0F && f.osh.size() == 1 && f.osh[0] == std::trunc(f.osh[0])) {
        // if outputScale == 1.0f and outputShift is interger, it can be further optimized
        //   x = clip2(round(x * inputScale + ish),c2lo,c2hi)*osc + osh
        //     = clip2(round(x * inputScale + ish),c2lo,c2hi) + osh
        //     = clip2(round(x * inputScale + ish) + osh, c2lo+osh,c2hi+osh)
        //     = clip2(round(x * inputScale + ish + osh), c2lo+osh,c2hi+osh)
        for (auto& v : f.ish) {
            v += f.osh[0];
        }
        for (auto& v : f.clo) {
            v += f.osh[0];
        }
        for (auto& v : f.chi) {
            v += f.osh[0];
        }
        f.osc.clear();
        f.osh.clear();
    }

    // we can save an additional eltwise linear for negligible shift
    if (all_of(1U, f.ish.size(), f.clo.size(), f.chi.size())) {
        auto range = (f.chi[0] - f.clo[0]);
        if (abs(f.ish[0]) < range * 0.00001F) {
            f.ish[0] = 0.0F;
        }
    }
}

// map FQ to oneDNN's attribuites & postOps
// equation:
//      y = clip2(round(x * inputScale + inputShift))*outputScale + outputShift
bool FakeQuantize::appendAttrPostOps(DnnlPostOpsComposerLegacy& dnnlpoc,
                                     bool isLastPostOp,
                                     dnnl::memory::data_type outDataType,
                                     bool allowBinary,
                                     bool doRounding) {
    DEBUG_LOG(getName(),
              ", isLastPostOp=",
              isLastPostOp,
              ", outDataType=",
              outDataType,
              ", allowBinary=",
              allowBinary,
              ", doRounding=",
              doRounding);
    DEBUG_LOG("\t ---- Original formula ----");
    DEBUG_LOG("\t    cropLow =[", printable(cropLow), "]");
    DEBUG_LOG("\t   cropHigh =[", printable(cropHigh), "]");
    DEBUG_LOG("\t inputScale =[", printable(inputScale), "]");
    DEBUG_LOG("\t inputShift =[", printable(inputShift), "]");
    DEBUG_LOG("\t outputScale=[", printable(outputScale), "]");
    DEBUG_LOG("\t outputShift=[", printable(outputShift), "]");

    const size_t bufferAlignment = 1;
    initializePostOpData(dnnlpoc.getOutputDims(), bufferAlignment, doRounding);

    auto& f = optimizedFormula;

    DEBUG_LOG("\t ---- Optimized formula ----");
    DEBUG_LOG("\t inputScale =[", printable(f.isc), "]");
    DEBUG_LOG("\t inputShift =[", printable(f.ish), "]");
    DEBUG_LOG("\t    cropLow =[", printable(f.clo), "]");
    DEBUG_LOG("\t   cropHigh =[", printable(f.chi), "]");
    DEBUG_LOG("\toutputScale =[", printable(f.osc), "]");
    DEBUG_LOG("\toutputShift =[", printable(f.osh), "]");

    // when FQ is last postOps and output data type is u8/s8
    // round & clip2 can be further optimized since saturation will be performed by oneDNN by default
    bool skipRoundClipOutputLinear = false;
    if (isLastPostOp && (levels == 256) && f.clo.size() == 1 && f.chi.size() == 1 && f.osc.empty() && f.osh.empty()) {
        if (outDataType == memory::data_type::u8 && f.clo[0] <= 0.0F && f.chi[0] >= 255.0F) {
            skipRoundClipOutputLinear = true;
        }
        if (outDataType == memory::data_type::s8 && f.clo[0] <= -128.0F && f.chi[0] >= 127.0F) {
            skipRoundClipOutputLinear = true;
        }
    }

    // return false before committing any change to DnnlPostOpsComposer
    if (!allowBinary) {
        if (f.ish.size() > 1) {
            return false;
        }
        if (!skipRoundClipOutputLinear) {
            if (f.clo.size() > 1 || f.chi.size() > 1) {
                return false;
            }
            if (f.osc.size() > 1 || f.osh.size() > 1) {
                return false;
            }
        }
    }

    if (!dnnlpoc.appendLinear(f.isc, f.ish, isLastPostOp && skipRoundClipOutputLinear, allowBinary)) {
        return false;
    }

    if (skipRoundClipOutputLinear) {
        return true;
    }

    if (doRounding) {
        dnnlpoc.appendRoundHTE();
    }
    dnnlpoc.appendClip(f.clo, f.chi);
    dnnlpoc.appendLinear(f.osc, f.osh, isLastPostOp, allowBinary);
    return true;
}

FakeQuantize::FakeQuantizeJitExecutor::FakeQuantizeJitExecutor([[maybe_unused]] const jit_quantize_params& _jqp) {
#if defined(OPENVINO_ARCH_X86_64)
    bool isBinarization = _jqp.op_type == Algorithm::FQBinarization;
    if (mayiuse(cpu::x64::avx512_core)) {
        if (isBinarization) {
            pKernel = std::make_unique<jit_uni_binarization_kernel<cpu::x64::avx512_core>>(_jqp);
        } else {
            pKernel = std::make_unique<jit_uni_quantization_kernel<cpu::x64::avx512_core>>(_jqp);
        }
    } else if (mayiuse(cpu::x64::avx2)) {
        if (isBinarization) {
            pKernel = std::make_unique<jit_uni_binarization_kernel<cpu::x64::avx2>>(_jqp);
        } else {
            pKernel = std::make_unique<jit_uni_quantization_kernel<cpu::x64::avx2>>(_jqp);
        }
    } else if (mayiuse(cpu::x64::sse41)) {
        if (isBinarization) {
            pKernel = std::make_unique<jit_uni_binarization_kernel<cpu::x64::sse41>>(_jqp);
        } else {
            pKernel = std::make_unique<jit_uni_quantization_kernel<cpu::x64::sse41>>(_jqp);
        }
    } else {
        OPENVINO_THROW("Can't create jit fake quantize kernel");
    }
    if (pKernel) {
        pKernel->create_ker();
    }
#endif
}

void FakeQuantize::FakeQuantizeJitExecutor::exec(const FakeQuantize& node) {
    OPENVINO_ASSERT(pKernel, "Can't execute, kernel for fake quantize node is not compiled");
    if (pKernel->jqp_.op_type == Algorithm::FQBinarization) {
        node.executeBinarization(pKernel);
    } else {
        node.executeQuantization(pKernel);
    }
}

bool FakeQuantize::created() const {
    return getType() == Type::FakeQuantize;
}

}  // namespace ov::intel_cpu::node
