// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include <iostream>
#include <sstream>

namespace ov {
namespace intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

template void RegPrinter::print<float, Xmm>(jit_generator &h, Xmm reg, const char *name);
template void RegPrinter::print<int, Xmm>(jit_generator &h, Xmm reg, const char *name);
template void RegPrinter::print<float, Ymm>(jit_generator &h, Ymm reg, const char *name);
template void RegPrinter::print<int, Ymm>(jit_generator &h, Ymm reg, const char *name);
template void RegPrinter::print<float, Zmm>(jit_generator &h, Zmm reg, const char *name);
template void RegPrinter::print<int, Zmm>(jit_generator &h, Zmm reg, const char *name);
template void RegPrinter::print<float, Reg64>(jit_generator &h, Reg64 reg, const char *name);
template void RegPrinter::print<int, Reg64>(jit_generator &h, Reg64 reg, const char *name);
template void RegPrinter::print<float, Reg32>(jit_generator &h, Reg32 reg, const char *name);
template void RegPrinter::print<int, Reg32>(jit_generator &h, Reg32 reg, const char *name);
template void RegPrinter::print<char, Reg16>(jit_generator &h, Reg16 reg, const char *name);
template void RegPrinter::print<unsigned char, Reg16>(jit_generator &h, Reg16 reg, const char *name);
template void RegPrinter::print<char, Reg8>(jit_generator &h, Reg8 reg, const char *name);
template void RegPrinter::print<unsigned char, Reg8>(jit_generator &h, Reg8 reg, const char *name);

void RegPrinter::print_reg_fp32(const char *name, int val) {
    std::stringstream ss;
    ss << name << ": " << *reinterpret_cast<float *>(&val) << std::endl;
    std::cout << ss.str();
}

template <typename T>
void RegPrinter::print_reg_integer(const char *name, T val) {
    std::stringstream ss;
    if (std::is_signed<T>::value) {
        ss << name << ": " << static_cast<int64_t>(val) << std::endl;
    } else {
        ss << name << ": " << static_cast<uint64_t>(val) << std::endl;
    }
    std::cout << ss.str();
}

template <typename PRC_T, size_t vlen>
void RegPrinter::print_vmm_prc(const char *name, PRC_T *ptr) {
    std::stringstream ss;
    ss << name << ": {" << ptr[0];
    for (size_t i = 1; i < vlen / sizeof(float); i++) {
        ss << ", " << ptr[i];
    }
    ss << "}" << std::endl;
    std::cout << ss.str();
}
template void RegPrinter::print_vmm_prc<float, 16>(const char *name, float *ptr);
template void RegPrinter::print_vmm_prc<float, 32>(const char *name, float *ptr);
template void RegPrinter::print_vmm_prc<float, 64>(const char *name, float *ptr);
template void RegPrinter::print_vmm_prc<int, 16>(const char *name, int *ptr);
template void RegPrinter::print_vmm_prc<int, 32>(const char *name, int *ptr);
template void RegPrinter::print_vmm_prc<int, 64>(const char *name, int *ptr);

template <typename Vmm>
struct vmm_traits{};

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
void RegPrinter::save_vmm(jit_generator &h) {
    h.sub(h.rsp, vmm_traits<T>::vmm_len * vmm_traits<T>::vmm_cnt);
    for (size_t i = 0; i < vmm_traits<T>::vmm_cnt; i++) {
        h.uni_vmovups(h.ptr[h.rsp + i * vmm_traits<T>::vmm_len], T(i));
    }
}

template <typename T>
void RegPrinter::restore_vmm(jit_generator &h) {
    for (size_t i = 0; i < vmm_traits<T>::vmm_cnt; i++) {
        h.uni_vmovups(T(i), h.ptr[h.rsp + i * vmm_traits<T>::vmm_len]);
    }
    h.add(h.rsp, vmm_traits<T>::vmm_len * vmm_traits<T>::vmm_cnt);
}

void RegPrinter::save_reg(jit_generator &h) {
    h.sub(h.rsp, reg_len * reg_cnt);
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(h.ptr[h.rsp + i * reg_len], Reg64(i));
    }
}

void RegPrinter::restore_reg(jit_generator &h) {
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(Reg64(i), h.ptr[h.rsp + i * reg_len]);
    }
    h.add(h.rsp, reg_len * reg_cnt);
}

void RegPrinter::preamble(jit_generator &h) {
    save_reg(h);
    mayiuse(cpu_isa_t::avx512_core) ? save_vmm<Zmm>(h) : (mayiuse(cpu_isa_t::avx2) ?
                   save_vmm<Ymm>(h) : save_vmm<Xmm>(h));
}

void RegPrinter::postamble(jit_generator &h) {
    mayiuse(cpu_isa_t::avx512_core) ? restore_vmm<Zmm>(h) : (mayiuse(cpu_isa_t::avx2) ?
                restore_vmm<Ymm>(h) : restore_vmm<Xmm>(h));
    restore_reg(h);
}

template <typename REG_T>
const char * RegPrinter::get_name(REG_T reg, const char *name) {
    const char *reg_name = reg.toString();

    if (name == nullptr) {
        return reg_name;
    } else {
        constexpr size_t len = 64;
        constexpr size_t aux_len = 3;
        static char full_name[len];

        size_t total_len = std::strlen(name) + std::strlen(reg_name) + aux_len + 1;
        if (total_len > len) {
           return reg_name;
        } else {
           snprintf(full_name, len, "%s | %s", name, reg_name);
           return full_name;
        }
    }
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_vmm(jit_generator &h, REG_T vmm, const char *name) {
    preamble(h);

    name = get_name(vmm, name);

    h.push(h.rax);
    h.push(abi_param1);
    h.push(abi_param2);
    {
        const int vlen = vmm.isZMM() ? 64 : (vmm.isYMM() ? 32 : 16);
        h.sub(h.rsp, vlen);
        h.uni_vmovups(h.ptr[h.rsp], vmm);

        h.mov(abi_param2, h.rsp);
        h.mov(abi_param1, reinterpret_cast<size_t>(name));
        if (vmm.isZMM()) {
            h.mov(h.rax, reinterpret_cast<size_t>(&print_vmm_prc<PRC_T, 64>));
        } else if (vmm.isYMM()) {
            h.mov(h.rax, reinterpret_cast<size_t>(&print_vmm_prc<PRC_T, 32>));
        } else {
            h.mov(h.rax, reinterpret_cast<size_t>(&print_vmm_prc<PRC_T, 16>));
        }
        h.call(h.rax);

        h.add(h.rsp, vlen);
    }

    h.pop(abi_param2);
    h.pop(abi_param1);
    h.pop(h.rax);

    postamble(h);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_reg(jit_generator &h, REG_T reg, const char *name) {
    preamble(h);

    name = get_name(reg, name);

    h.push(h.rax);
    h.push(abi_param1);
    h.push(abi_param2);
    {
        h.mov(abi_param2, reg);
        h.mov(abi_param1, reinterpret_cast<size_t>(name));
        if (std::is_floating_point<PRC_T>::value)
            h.mov(h.rax, reinterpret_cast<size_t>(&print_reg_fp32));
        else
            h.mov(h.rax, reinterpret_cast<size_t>(&print_reg_integer<PRC_T>));
        h.call(h.rax);
    }

    h.pop(abi_param2);
    h.pop(abi_param1);
    h.pop(h.rax);

    postamble(h);
}

}   // namespace intel_cpu
}   // namespace ov
