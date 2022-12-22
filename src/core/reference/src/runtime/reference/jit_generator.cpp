// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_generator.hpp"

#include <xbyak/xbyak_util.h>

#include "ngraph/type/float16.hpp"

namespace ngraph {
namespace runtime {
namespace jit {
using namespace Xbyak;

#ifdef XBYAK64
static const Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#    ifdef _WIN32
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
#    endif
};

#    ifdef _WIN32
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX), abi_param2(Xbyak::Operand::RDX),
    abi_param3(Xbyak::Operand::R8), abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#    else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI), abi_param2(Xbyak::Operand::RSI),
    abi_param3(Xbyak::Operand::RDX), abi_param4(Xbyak::Operand::RCX), abi_param5(Xbyak::Operand::R8),
    abi_param6(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RCX);
#    endif
#endif

const size_t Generator::num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

const Xbyak::Reg64 Generator::param = abi_param1;

bool Generator::mayiuse(const cpu_isa_t cpu_isa) {
    static Xbyak::util::Cpu cpu;

    using namespace Xbyak::util;

    switch (cpu_isa) {
    case sse42:
        return cpu.has(Cpu::tSSE42);
    case avx:
        return cpu.has(Cpu::tAVX);
    case avx2:
        return cpu.has(Cpu::tAVX2);
    case avx512_common:
        return cpu.has(Cpu::tAVX512F);
    case avx512_core:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) && cpu.has(Cpu::tAVX512VL) &&
               cpu.has(Cpu::tAVX512DQ) && cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
        return cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) && cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
        return mayiuse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) && cpu.has(Cpu::tAVX512_4VNNIW);
    case avx512_core_bf16:
        return mayiuse(avx512_core_vnni) && cpu.has(Cpu::tAVX512_BF16);
    case avx512_vpopcnt:
        return true && cpu.has(Cpu::tAVX512_VPOPCNTDQ);
    case fp16:
        return cpu.has(Cpu::tF16C);
    case isa_any:
        return true;
    }
    return false;
}

bool Generator::is_x64() {
    return sizeof(void*) == 8;
}
Generator::Generator(void* code_ptr, size_t code_size)
    : Xbyak::CodeGenerator(code_size, code_ptr),
      size_of_abi_save_regs(num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * xmm_len),
      reg_EVEX_max_8b_offt(rbp) {}

void Generator::preamble() {
    if (xmm_to_preserve) {
        sub(rsp, xmm_to_preserve * xmm_len);
        for (size_t i = 0; i < xmm_to_preserve; ++i)
            movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(static_cast<int>(xmm_to_preserve_start + i)));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
        push(Xbyak::Reg64(abi_save_gpr_regs[i]));
    if (mayiuse(avx512_common)) {
        mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
    }
}

void Generator::postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
        pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    if (xmm_to_preserve) {
        for (size_t i = 0; i < xmm_to_preserve; ++i)
            movdqu(Xbyak::Xmm(static_cast<int>(xmm_to_preserve_start + i)), ptr[rsp + i * xmm_len]);
        add(rsp, xmm_to_preserve * xmm_len);
    }
    if (mayiuse(avx) && !mayiuse(avx512_mic))
        vzeroupper();
    ret();
}

void Generator::foreach (const Xbyak::Reg64& idx,
                         size_t step,
                         const Xbyak::Reg64& end,
                         std::function<void(const Xbyak::Reg64&)> && fn) {
    Label loop, exit;

    L(loop);
    cmp(idx, end);
    jge(exit);

    fn(idx);

    add(idx, static_cast<uint32_t>(step));
    jmp(loop);
    L(exit);
}

template <>
void Generator::copy<uint8_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15b, byte[src + idx * sizeof(uint8_t)]);
        mov(byte[dst + idx * sizeof(uint8_t)], r15b);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<int8_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15b, byte[src + idx * sizeof(int8_t)]);
        mov(byte[dst + idx * sizeof(int8_t)], r15b);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<uint16_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15w, word[src + idx * sizeof(uint16_t)]);
        mov(word[dst + idx * sizeof(uint16_t)], r15w);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<uint32_t>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    push(rsi);
    push(r15);

    xor_(rsi, rsi);

    foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
        mov(r15d, dword[src + idx * sizeof(uint32_t)]);
        mov(dword[dst + idx * sizeof(uint32_t)], r15d);
    })
        ;

    pop(r15);
    pop(rsi);
}

template <>
void Generator::copy<float16>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    copy<uint16_t>(dst, src, size);
}

template <>
void Generator::copy<float>(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
    copy<uint32_t>(dst, src, size);
}
}  // namespace jit
}  // namespace runtime
}  // namespace ngraph
