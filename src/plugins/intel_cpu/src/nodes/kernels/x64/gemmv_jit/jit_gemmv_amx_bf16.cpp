// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_amx_bf16.hpp"

#include "openvino/core/except.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {
constexpr int64_t k_bytes_per_row_a = 128; // 64 bf16 values
constexpr int64_t k_bytes_per_row_b = 32;  // 16 bf16 values
constexpr int64_t k_bytes_per_row_c = 64;  // 16 fp32 values
} // namespace

jit_amx_gemmv_bf16_t::jit_amx_gemmv_bf16_t()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_amx_gemmv_bf16_t",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_amx_gemmv_bf16 kernel");
    }
    kernel_ = reinterpret_cast<kernel_fn>(jit_ker());
}

const jit_amx_gemmv_bf16_t& jit_amx_gemmv_bf16_t::instance() {
    static const jit_amx_gemmv_bf16_t jit;
    return jit;
}

void jit_amx_gemmv_bf16_t::generate() {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif

#if defined(_WIN32)
    const Reg64 reg_param = rcx;
#else
    const Reg64 reg_param = rdi;
#endif
    const Reg64 reg_a_ptr = r8;
    const Reg64 reg_b_ptr = r9;
    const Reg64 reg_c_ptr = r10;
    const Reg64 reg_a_stride = r11;
    const Reg64 reg_b_stride = r12;
    const Reg64 reg_k_blocks = r13;
    const Reg64 reg_stride_a = rbx;
    const Reg64 reg_stride_b = rcx;
    const Reg64 reg_stride_c = r14;

#if defined(_WIN32)
    push(rsi);
#endif
    push(rbx);
    push(rcx);
    push(r12);
    push(r13);
    push(r14);

    mov(reg_a_ptr, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, a_ptr)]);
    mov(reg_b_ptr, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, b_ptr)]);
    mov(reg_c_ptr, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, c_out)]);
    mov(reg_a_stride, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, a_tile_bytes)]);
    mov(reg_b_stride, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, b_group_bytes)]);
    mov(reg_k_blocks, qword[reg_param + offsetof(jit_amx_gemmv_bf16_call_args, k_blocks)]);
    mov(reg_stride_a, k_bytes_per_row_a);
    mov(reg_stride_b, k_bytes_per_row_b);

    tilezero(Tmm(0));

    Label label_loop;
    Label label_done;

    cmp(reg_k_blocks, 0);
    jz(label_done);

    L(label_loop);
    {
        tileloadd(Tmm(1), ptr[reg_a_ptr + reg_stride_a]);
        tileloadd(Tmm(2), ptr[reg_b_ptr + reg_stride_b]);
        tdpbf16ps(Tmm(0), Tmm(1), Tmm(2));
        add(reg_a_ptr, reg_a_stride);
        add(reg_b_ptr, reg_b_stride);
        dec(reg_k_blocks);
        jnz(label_loop);
    }

    L(label_done);
    mov(reg_stride_c, k_bytes_per_row_c);
    tilestored(ptr[reg_c_ptr + reg_stride_c], Tmm(0));

    pop(r14);
    pop(r13);
    pop(r12);
    pop(rcx);
    pop(rbx);
#if defined(_WIN32)
    pop(rsi);
#endif
    ret();
}

} // namespace ov::intel_cpu::x64::gemmv_jit
