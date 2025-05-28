// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_matmul_small.hpp"

#include <cpu/x64/xbyak/xbyak.h>

#include <cassert>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

#define GET_OFF(field) offsetof(jit_matmul_small_call_args, field)

namespace ov::intel_cpu {

template <cpu::x64::cpu_isa_t isa>
void jit_uni_matmul_small_kernel_f32<isa>::generate() {
    this->preamble();

    mov(reg_input1, ptr[reg_params + GET_OFF(input1)]);
    mov(reg_input2, ptr[reg_params + GET_OFF(input2)]);
    mov(reg_out, ptr[reg_params + GET_OFF(output)]);
    mov(reg_work_amount, ptr[reg_params + GET_OFF(B)]);
    if (jcp_.M > 2 || jcp_.N > 2 || jcp_.K > 2) {
        assert("matmul_small_kernel only support M/N/K smaller than 3.");
    }

    Xbyak::Label loop_label;
    Xbyak::Label loop_end_label;
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
                uni_vmovss(ptr[reg_out], vmm_output[m * jcp_.N + n]);
                add(reg_out, sizeof(float));
            }
        }

        sub(reg_work_amount, 1);
        jmp(loop_label, T_NEAR);
    }
    L(loop_end_label);

    this->postamble();
}

template struct jit_uni_matmul_small_kernel_f32<cpu::x64::sse41>;
template struct jit_uni_matmul_small_kernel_f32<cpu::x64::avx2>;
template struct jit_uni_matmul_small_kernel_f32<cpu::x64::avx512_core>;

}  // namespace ov::intel_cpu