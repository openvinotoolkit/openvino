// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS

#    include "debug_capabilities.hpp"

#    include <xbyak/xbyak.h>

#    include <cpu/x64/cpu_isa_traits.hpp>
#    include <cpu/x64/jit_generator.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <iostream>
#    include <sstream>

#    include "emitters/plugin/common/debug_utils.hpp"

namespace ov::intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

template void RegPrinter::print<float, Xmm>(jit_generator_t& h, Xmm reg, const char* name);
template void RegPrinter::print<int, Xmm>(jit_generator_t& h, Xmm reg, const char* name);
template void RegPrinter::print<float, Ymm>(jit_generator_t& h, Ymm reg, const char* name);
template void RegPrinter::print<int, Ymm>(jit_generator_t& h, Ymm reg, const char* name);
template void RegPrinter::print<float, Zmm>(jit_generator_t& h, Zmm reg, const char* name);
template void RegPrinter::print<int, Zmm>(jit_generator_t& h, Zmm reg, const char* name);
template void RegPrinter::print<float, Reg64>(jit_generator_t& h, Reg64 reg, const char* name);
template void RegPrinter::print<int, Reg64>(jit_generator_t& h, Reg64 reg, const char* name);
template void RegPrinter::print<float, Reg32>(jit_generator_t& h, Reg32 reg, const char* name);
template void RegPrinter::print<int, Reg32>(jit_generator_t& h, Reg32 reg, const char* name);
template void RegPrinter::print<char, Reg16>(jit_generator_t& h, Reg16 reg, const char* name);
template void RegPrinter::print<unsigned char, Reg16>(jit_generator_t& h, Reg16 reg, const char* name);
template void RegPrinter::print<char, Reg8>(jit_generator_t& h, Reg8 reg, const char* name);
template void RegPrinter::print<unsigned char, Reg8>(jit_generator_t& h, Reg8 reg, const char* name);

template <typename Vmm>
struct vmm_traits {};

template <>
struct vmm_traits<Xmm> {
    static constexpr size_t vmm_len = 16;
    static constexpr size_t vmm_cnt = 16;
};

template <>
struct vmm_traits<Ymm> {
    static constexpr size_t vmm_len = 32;
    static constexpr size_t vmm_cnt = 16;
};

template <>
struct vmm_traits<Zmm> {
    static constexpr size_t vmm_len = 64;
    static constexpr size_t vmm_cnt = 32;
};

template <typename T>
void RegPrinter::save_vmm(jit_generator_t& h) {
    h.sub(h.rsp, vmm_traits<T>::vmm_len * vmm_traits<T>::vmm_cnt);
    for (size_t i = 0; i < vmm_traits<T>::vmm_cnt; i++) {
        h.uni_vmovups(h.ptr[h.rsp + i * vmm_traits<T>::vmm_len], T(i));
    }
}

template <typename T>
void RegPrinter::restore_vmm(jit_generator_t& h) {
    for (size_t i = 0; i < vmm_traits<T>::vmm_cnt; i++) {
        h.uni_vmovups(T(i), h.ptr[h.rsp + i * vmm_traits<T>::vmm_len]);
    }
    h.add(h.rsp, vmm_traits<T>::vmm_len * vmm_traits<T>::vmm_cnt);
}

void RegPrinter::save_reg(jit_generator_t& h) {
    h.sub(h.rsp, reg_len * reg_cnt);
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(h.ptr[h.rsp + i * reg_len], Reg64(i));
    }
}

void RegPrinter::restore_reg(jit_generator_t& h) {
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(Reg64(i), h.ptr[h.rsp + i * reg_len]);
    }
    h.add(h.rsp, reg_len * reg_cnt);
}

void RegPrinter::preamble(jit_generator_t& h) {
    save_reg(h);
    if (mayiuse(cpu_isa_t::avx512_core)) {
        save_vmm<Zmm>(h);
    } else if (mayiuse(cpu_isa_t::avx2)) {
        save_vmm<Ymm>(h);
    } else {
        save_vmm<Xmm>(h);
    }
}

void RegPrinter::postamble(jit_generator_t& h) {
    if (mayiuse(cpu_isa_t::avx512_core)) {
        restore_vmm<Zmm>(h);
    } else if (mayiuse(cpu_isa_t::avx2)) {
        restore_vmm<Ymm>(h);
    } else {
        restore_vmm<Xmm>(h);
    }
    restore_reg(h);
}

// ABI requires 16-bype stack alignment before a call
void RegPrinter::align_rsp(jit_generator_t& h) {
    constexpr int alignment = 16;
    h.mov(h.r15, h.rsp);
    h.and_(h.rsp, ~(alignment - 1));
}

void RegPrinter::restore_rsp(jit_generator_t& h) {
    h.mov(h.rsp, h.r15);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_vmm(jit_generator_t& h, REG_T vmm, const char* name) {
    preamble(h);

    h.push(h.rax);
    h.push(abi_param1);
    h.push(abi_param2);
    h.push(abi_param3);
    {
        int vlen = [&]() {
            if (vmm.isZMM()) {
                return 64;
            }
            if (vmm.isYMM()) {
                return 32;
            }
            return 16;
        }();
        h.sub(h.rsp, vlen);
        h.uni_vmovups(h.ptr[h.rsp], vmm);

        h.mov(abi_param3, h.rsp);
        h.mov(abi_param2, reinterpret_cast<size_t>(vmm.toString()));
        h.mov(abi_param1, reinterpret_cast<size_t>(name));
        if (vmm.isZMM()) {
            auto p = &ov::intel_cpu::debug_utils::print_vmm_prc<PRC_T, 64>;
            h.mov(h.rax, reinterpret_cast<size_t>(p));
        } else if (vmm.isYMM()) {
            auto p = &ov::intel_cpu::debug_utils::print_vmm_prc<PRC_T, 32>;
            h.mov(h.rax, reinterpret_cast<size_t>(p));
        } else {
            auto p = &ov::intel_cpu::debug_utils::print_vmm_prc<PRC_T, 16>;
            h.mov(h.rax, reinterpret_cast<size_t>(p));
        }
        align_rsp(h);
        h.call(h.rax);
        restore_rsp(h);

        h.add(h.rsp, vlen);
    }

    h.pop(abi_param3);
    h.pop(abi_param2);
    h.pop(abi_param1);
    h.pop(h.rax);

    postamble(h);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_reg(jit_generator_t& h, REG_T reg, const char* name) {
    preamble(h);

    h.push(h.rax);
    h.push(abi_param1);
    h.push(abi_param2);
    h.push(abi_param3);
    {
        const int rlen = reg.getBit() / 8;
        h.sub(h.rsp, rlen);
        h.mov(h.ptr[h.rsp], reg);

        h.mov(abi_param3, h.rsp);
        h.mov(abi_param2, reinterpret_cast<size_t>(reg.toString()));
        h.mov(abi_param1, reinterpret_cast<size_t>(name));
        auto p = &ov::intel_cpu::debug_utils::print_reg_prc<PRC_T>;
        h.mov(h.rax, reinterpret_cast<size_t>(p));
        align_rsp(h);
        h.call(h.rax);
        restore_rsp(h);

        h.add(h.rsp, rlen);
    }

    h.pop(abi_param3);
    h.pop(abi_param2);
    h.pop(abi_param1);
    h.pop(h.rax);

    postamble(h);
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
