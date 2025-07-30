// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_matmul_small.hpp"

#include <xbyak/xbyak.h>

#include <cassert>
#include <common/c_types_map.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

#include "cpu/x64/injectors/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_quantization_injector.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

#define GET_OFF(field) offsetof(jit_matmul_small_call_args, field)

namespace ov::intel_cpu {

template <cpu::x64::cpu_isa_t isa>
void jit_uni_matmul_small_kernel_f32<isa>::generate() {
    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_t<isa>>(this,
                                                                                          post_op.eltwise.alg,
                                                                                          post_op.eltwise.alpha,
                                                                                          post_op.eltwise.beta,
                                                                                          1.f,
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

    this->preamble();

    mov(reg_input1, ptr[reg_params + GET_OFF(input1)]);
    mov(reg_input2, ptr[reg_params + GET_OFF(input2)]);
    mov(reg_out, ptr[reg_params + GET_OFF(output)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(B)]);
    if (jcp_.M > 2 || jcp_.N > 2 || jcp_.K > 2) {
        assert("matmul_small_kernel only support M/N/K smaller than 3.");
    }

    if (attr_.post_ops_.len() != 0) {
        mov(reg_post_ops_data, ptr[reg_params + GET_OFF(post_op_data)]);
        mov(reg_oc, ptr[reg_params + GET_OFF(oc)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);
    }

    Xbyak::Label loop_label;
    Xbyak::Label loop_end_label;
    Xbyak::Label offset_reset_label;
    L(loop_label);
    {
        cmp(reg_work_amount, 1);
        jl(loop_end_label, T_NEAR);

        // loop unrolling and register utilization in each batch
        // load
        for (size_t m = 0; m < jcp_.M; m++) {
            for (size_t k = 0; k < jcp_.K; k++) {
                uni_vmovss(vmm_input1[m * jcp_.K + k], ptr[reg_input1]);
                add(reg_input1, sizeof(float));
            }
        }
        for (size_t k = 0; k < jcp_.K; k++) {
            for (size_t n = 0; n < jcp_.N; n++) {
                uni_vmovss(vmm_input2[k * jcp_.N + n], ptr[reg_input2]);
                add(reg_input2, sizeof(float));
            }
        }

        for (size_t m = 0; m < jcp_.M; m++) {
            for (size_t n = 0; n < jcp_.N; n++) {
                uni_vpxor(vmm_output[m * jcp_.N + n], vmm_output[m * jcp_.N + n], vmm_output[m * jcp_.N + n]);
            }
        }
        // outer most K to reduce RAW dependency.
        for (size_t k = 0; k < jcp_.K; k++) {
            for (size_t m = 0; m < jcp_.M; m++) {
                for (size_t n = 0; n < jcp_.N; n++) {
                    uni_vfmadd231ps(vmm_output[m * jcp_.N + n], vmm_input1[m * jcp_.K + k], vmm_input2[k * jcp_.N + n]);
                }
            }
        }

        // store
        for (size_t m = 0; m < jcp_.M; m++) {
            for (size_t n = 0; n < jcp_.N; n++) {
                if (attr_.post_ops_.len() != 0) {
                    apply_post_ops(element::f32, vmm_output[m * jcp_.N + n].getIdx(), true);
                }
                uni_vmovss(ptr[reg_out], vmm_output[m * jcp_.N + n]);
                add(reg_out, sizeof(float));
            }
        }

        if (attr_.post_ops_.len() != 0) {
            add(reg_oc_off, 1);
            cmp(reg_oc_off, reg_oc);
            jl(offset_reset_label, T_NEAR);
            mov(reg_oc_off, 0);
            L(offset_reset_label);
        }

        sub(reg_work_amount, 1);
        jmp(loop_label, T_NEAR);
    }
    L(loop_end_label);

    this->postamble();

    for (auto& inj : eltwise_injectors) {
        inj->prepare_table();
    }
}

// apply post ops on same channel. reg_oc_off is reset to start if reach oc.
template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_uni_matmul_small_kernel_f32<isa>::apply_post_ops(ov::element::Type dst_prc,
                                                          size_t vmm_idx,
                                                          bool is_broadcast) {
    const auto& p = attr_.post_ops_;
    int eltwise_inj_idx = 0;
    int depthwise_inj_idx = 0;
    int quantization_inj_idx = 0;
    int post_ops_data_offset = 0;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_idx, vmm_idx + 1);
            eltwise_inj_idx++;
        } else if (post_op.is_depthwise()) {
            mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
            add(reg_d_weights, reg_oc_off);

            // weight and bias is padded. scalar as vector.
            depthwise_injectors[depthwise_inj_idx]->compute_vector_range(vmm_idx,
                                                                         vmm_idx + 1,
                                                                         reg_d_weights,
                                                                         reg_d_weights,
                                                                         is_broadcast);

            post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
            depthwise_inj_idx++;
        } else if (post_op.is_quantization()) {
            bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
            bool do_rounding = do_dequantization || dst_prc == ov::element::f32 || i != p.len() - 1;

            quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset,
                                                                         reg_oc_off);
            quantization_injectors[quantization_inj_idx]->compute_crop(vmm_idx, vmm_idx + 1, 0, 0, is_broadcast);

            quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(
                reg_post_ops_data + post_ops_data_offset,
                reg_oc_off);
            quantization_injectors[quantization_inj_idx]
                ->compute_input_scale_shift(vmm_idx, vmm_idx + 1, 0, do_rounding, 0, is_broadcast);

            if (do_dequantization) {
                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(
                    reg_post_ops_data + post_ops_data_offset,
                    reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(vmm_idx,
                                                                                         vmm_idx + 1,
                                                                                         0,
                                                                                         0,
                                                                                         is_broadcast);
            }

            post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
            quantization_inj_idx++;
        }
    }
}

template struct jit_uni_matmul_small_kernel_f32<cpu::x64::sse41>;
template struct jit_uni_matmul_small_kernel_f32<cpu::x64::avx2>;
template struct jit_uni_matmul_small_kernel_f32<cpu::x64::avx512_core>;

}  // namespace ov::intel_cpu
