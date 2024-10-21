// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined _WIN32 && !defined NOMINMAX
#    define NOMINMAX
#endif

#include <xbyak/xbyak.h>

#include <functional>

namespace ov {
namespace reference {
namespace jit {
static const Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
#endif
};

#ifdef _WIN32
#    define abi_param1 Xbyak::Reg64(Xbyak::Operand::RCX)  // RCX
#else
#    define abi_param1 Xbyak::Reg64(Xbyak::Operand::RDI)  // RDI
#endif

typedef enum {
    isa_any,
    sse42,
    avx,
    avx2,
    avx512_common,
    avx512_core,
    avx512_core_vnni,
    avx512_mic,
    avx512_mic_4ops,
    avx512_core_bf16,
    avx512_vpopcnt,
    fp16,
    pclmulqdq,
    vpclmulqdq
} cpu_isa_t;

class Generator : public Xbyak::CodeGenerator {
#ifdef _WIN32
    static constexpr size_t xmm_to_preserve_start = 6llu;
    static constexpr size_t xmm_to_preserve = 10llu;
#else
    static constexpr size_t xmm_to_preserve_start = 0lu;
    static constexpr size_t xmm_to_preserve = 0lu;
#endif

    static const size_t num_abi_save_gpr_regs = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);
    const size_t size_of_abi_save_regs;

    const Xbyak::Reg64 reg_EVEX_max_8b_offt;
    static constexpr int EVEX_max_8b_offt = 0x200;
    size_t m_vlen = ymm_len;

public:
    static constexpr size_t xmm_len = 16lu;
    static constexpr size_t ymm_len = 32lu;
    static constexpr size_t zmm_len = 64lu;

    const Xbyak::Reg64 param = abi_param1;

    static bool mayiuse(const cpu_isa_t cpu_isa);
    static bool is_x64();

    Generator(cpu_isa_t isa = avx2, void* code_ptr = nullptr, size_t code_size = 16lu * 1024lu);
    void preamble();
    void postamble();

    void foreach (const Xbyak::Reg64& idx,
                  size_t step,
                  const Xbyak::Reg64& end,
                  std::function<void(const Xbyak::Reg64&)> && fn);

    template <typename T>
    void copy(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size);

    size_t get_vlen() {
        return m_vlen;
    }
};

}  // namespace jit
}  // namespace reference
}  // namespace ov
