// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_amx_int8.hpp"

#include "openvino/core/except.hpp"

namespace ov::intel_cpu::x64::gemmv_jit {

namespace {

// A tile: 16 rows x 64 columns (weights for 16 output lanes)
constexpr int64_t k_bytes_per_row_a = 64;
constexpr int64_t k_rows_m = 16;
constexpr int64_t k_tile_bytes_a = k_bytes_per_row_a * k_rows_m;

// B tile (weights): 16 rows x 4 columns (VNNI, colsb=4)
constexpr int64_t k_bytes_per_row_b = 4;
constexpr int64_t k_rows_k = 16;
constexpr int64_t k_tile_bytes_b = k_bytes_per_row_b * k_rows_k;

// C tile: 16 rows x 4 bytes stride (one dword per row)
constexpr int64_t k_bytes_per_row_c = 4;

} // namespace

jit_amx_gemmv_int8_t::jit_amx_gemmv_int8_t()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_amx_gemmv_int8_t",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_amx_gemmv_int8 kernel");
    }
    kernel_ = reinterpret_cast<kernel_fn>(jit_ker());
    code_ptr_ = jit_ker();
    code_size_ = getSize();
    dnnl::impl::cpu::jit_utils::register_jit_code(code_ptr_, code_size_, "/gemmv:amx_int8", __FILE__);
}

const jit_amx_gemmv_int8_t& jit_amx_gemmv_int8_t::instance() {
    static const jit_amx_gemmv_int8_t jit;
    return jit;
}

void jit_amx_gemmv_int8_t::generate() {
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
    const Reg64 reg_palette = rbx;
    const Reg64 reg_a_ptr = r8;
    const Reg64 reg_b_ptr = r9;
    const Reg64 reg_c_ptr = r10;
    const Reg64 reg_a_stride = r11;
    const Reg64 reg_b_stride = r12;
    const Reg64 reg_blocks = r13;
    const Reg64 reg_tmp = r14;

#if defined(_WIN32)
    push(rsi);
#endif
    push(rbx);
    push(rbp);
    push(r11);
    push(r12);
    push(r13);
    push(r14);

    mov(reg_palette, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, tilecfg)]);
    mov(reg_a_ptr, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, a_ptr)]);
    mov(reg_b_ptr, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, b_ptr)]);
    mov(reg_c_ptr, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, c_out)]);
    mov(reg_blocks, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, k_blocks)]);
    mov(reg_a_stride, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, a_tile_bytes)]);
    mov(reg_b_stride, qword[reg_param + offsetof(jit_amx_gemmv_int8_call_args, b_group_bytes)]);

    // Configure palette locally so AMX instructions never depend on caller state.
    ldtilecfg(ptr[reg_palette]);
    tilezero(Tmm(0));
    xor_(reg_tmp, reg_tmp);

    Label label_loop;
    Label label_store;

    test(reg_blocks, reg_blocks);
    jz(label_store);

    L(label_loop);
    // AMX semantics: tdpbusd dst, src1(u8), src2(s8)
    // Palette: T1 (cols=64) holds W (s8), T2 (cols=4) holds X (u8)
    tileloadd(Tmm(1), ptr[reg_a_ptr + reg_tmp]); // W (s8), rows=M, colsb=64
    tileloadd(Tmm(2), ptr[reg_b_ptr + reg_tmp]); // X (u8), rows=rd_block/rd_step, colsb=4
    tdpbsud(Tmm(0), Tmm(1), Tmm(2));
    add(reg_a_ptr, reg_a_stride);
    add(reg_b_ptr, reg_b_stride);
    dec(reg_blocks);
    jnz(label_loop);

    L(label_store);
    tilestored(ptr[reg_c_ptr + reg_tmp], Tmm(0));

    tilerelease();
    pop(r14);
    pop(r13);
    pop(r12);
    pop(r11);
    pop(rbp);
    pop(rbx);
#if defined(_WIN32)
    pop(rsi);
#endif
    ret();
}

} // namespace ov::intel_cpu::x64::gemmv_jit
