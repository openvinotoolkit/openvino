// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_matmul_small.hpp"

#include <cpu/x64/xbyak/xbyak.h>

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
const static size_t BLOCK_SIZE = 2;

namespace ov::intel_cpu {

template <cpu::x64::cpu_isa_t isa>
void jit_uni_matmul_small_kernel_f32<isa>::generate() {
    const auto& p = attr_.post_ops_;
    for (int i = 0; i < p.len(); i++) {
        auto& post_op = p.entry_[i];
        if (post_op.is_eltwise()) {
            eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector<isa>>(this,
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

        const size_t& m_block_num = div_up(jcp_.M, BLOCK_SIZE);
        const size_t& n_block_num = div_up(jcp_.N, BLOCK_SIZE);

        for (size_t m_blk = 0; m_blk < m_block_num; m_blk++) {
            const bool is_m_tail = (jcp_.M - m_blk * BLOCK_SIZE < BLOCK_SIZE);
            const size_t& m = is_m_tail ? 1 : 2;
            const size_t& in1_offset = m_blk * BLOCK_SIZE * jcp_.K * sizeof(float);
            for (size_t n_blk = 0; n_blk < n_block_num; n_blk++) {
                const bool is_n_tail = (jcp_.N - n_blk * BLOCK_SIZE < BLOCK_SIZE);
                const size_t& n = is_n_tail ? 1 : 2;
                const size_t& in2_offset = n_blk * BLOCK_SIZE * sizeof(float);
                const size_t& out_offset = (m_blk * BLOCK_SIZE * jcp_.N + n_blk * BLOCK_SIZE) * sizeof(float);
                ukernel_2k2(m, n, jcp_.K, in1_offset, in2_offset, out_offset, jcp_.K, jcp_.N, jcp_.N);
            }
        }

        if (attr_.post_ops_.len() != 0) {
            add(reg_oc_off, 1);
            cmp(reg_oc_off, reg_oc);
            jl(offset_reset_label, T_NEAR);
            mov(reg_oc_off, 0);
            L(offset_reset_label);
        }

        add(reg_input1, jcp_.M * jcp_.K * sizeof(float));
        add(reg_input2, jcp_.K * jcp_.N * sizeof(float));
        add(reg_out, jcp_.M * jcp_.N * sizeof(float));

        sub(reg_work_amount, 1);
        jmp(loop_label, T_NEAR);
    }
    L(loop_end_label);

    this->postamble();

    for (auto& inj : eltwise_injectors) {
        inj->prepare_table();
    }
}

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
void jit_uni_matmul_small_kernel_f32<isa>::ukernel_2k2(const size_t& M,
                                                       const size_t& N,
                                                       const size_t& K_full_size,
                                                       const size_t& offset_in1,
                                                       const size_t& offset_in2,
                                                       const size_t& offset_out,
                                                       const size_t& lda,
                                                       const size_t& ldb,
                                                       const size_t& ldc) {
    if (M > 2 || N > 2) {
        assert("matmul_small_kernel::ukernel_2k2 only support M/N smaller or equal to 2.");
    }
    const size_t& k_block_num = div_up(K_full_size, BLOCK_SIZE);
    for (size_t i = 0; i < k_block_num; i++) {
        const bool is_k_tail = (K_full_size - i * BLOCK_SIZE < BLOCK_SIZE);
        const size_t& K = is_k_tail ? 1 : 2;
        const float& beta = (i == 0) ? 0 : 1;
        const size_t& oc_in1 = offset_in1 + i * BLOCK_SIZE * sizeof(float);
        const size_t& oc_in2 = offset_in2 + i * BLOCK_SIZE * ldb * sizeof(float);

        for (size_t m = 0; m < M; m++) {
            const size_t& oc_inc_m = m * lda * sizeof(float);
            for (size_t k = 0; k < K; k++) {
                const size_t& oc_inc_k = k * sizeof(float);
                uni_vmovss(vmm_input1[m * K + k], ptr[reg_input1 + oc_in1 + oc_inc_m + oc_inc_k]);
            }
        }
        for (size_t k = 0; k < K; k++) {
            const size_t& oc_inc_k = k * ldb * sizeof(float);
            for (size_t n = 0; n < N; n++) {
                const size_t& oc_inc_n = n * sizeof(float);
                uni_vmovss(vmm_input2[k * N + n], ptr[reg_input2 + oc_in2 + oc_inc_k + oc_inc_n]);
            }
        }

        if (beta == 0) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    uni_vpxor(vmm_output[m * N + n], vmm_output[m * N + n], vmm_output[m * N + n]);
                }
            }
        }

        // outer most K to reduce RAW dependency.
        for (size_t k = 0; k < K; k++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t n = 0; n < N; n++) {
                    uni_vfmadd231ps(vmm_output[m * N + n], vmm_input1[m * K + k], vmm_input2[k * N + n]);
                }
            }
        }
    }

    // store
    for (size_t m = 0; m < M; m++) {
        const size_t& oc_inc_m = m * ldc * sizeof(float);
        for (size_t n = 0; n < N; n++) {
            const size_t& oc_inc_n = n * sizeof(float);
            if (attr_.post_ops_.len() != 0) {
                apply_post_ops(element::f32, vmm_output[m * N + n].getIdx(), true);
            }
            uni_vmovss(ptr[reg_out + offset_out + oc_inc_m + oc_inc_n], vmm_output[m * N + n]);
        }
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