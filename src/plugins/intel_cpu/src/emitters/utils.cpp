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

template <>
void RegPrinter::print<float, Xmm>(jit_generator &h, Xmm reg, const char *name) {
    print_vmm<float, Xmm>(h, reg, name);
}

template <>
void RegPrinter::print<int, Xmm>(jit_generator &h, Xmm reg, const char *name) {
    print_vmm<int, Xmm>(h, reg, name);
}

template <>
void RegPrinter::print<float, Ymm>(jit_generator &h, Ymm reg, const char *name) {
    print_vmm<float, Ymm>(h, reg, name);
}

template <>
void RegPrinter::print<int, Ymm>(jit_generator &h, Ymm reg, const char *name) {
    print_vmm<int, Ymm>(h, reg, name);
}

template <>
void RegPrinter::print<float, Zmm>(jit_generator &h, Zmm reg, const char *name) {
    print_vmm<float, Zmm>(h, reg, name);
}

template <>
void RegPrinter::print<int, Zmm>(jit_generator &h, Zmm reg, const char *name) {
    print_vmm<int, Zmm>(h, reg, name);
}

template <>
void RegPrinter::print<float, Reg64>(jit_generator &h, Reg64 reg, const char *name) {
    print_reg<float, Reg64>(h, reg, name);
}

template <>
void RegPrinter::print<int, Reg64>(jit_generator &h, Reg64 reg, const char *name) {
    print_reg<int, Reg64>(h, reg, name);
}

template <>
void RegPrinter::print<float, Reg32>(jit_generator &h, Reg32 reg, const char *name) {
    print_reg<float, Reg32>(h, reg, name);
}

template <>
void RegPrinter::print<int, Reg32>(jit_generator &h, Reg32 reg, const char *name) {
    print_reg<int, Reg32>(h, reg, name);
}

template <>
void RegPrinter::print<char, Reg16>(jit_generator &h, Reg16 reg, const char *name) {
    print_reg<char, Reg16>(h, reg, name);
}

template <>
void RegPrinter::print<unsigned char, Reg16>(jit_generator &h, Reg16 reg, const char *name) {
    print_reg<unsigned char, Reg16>(h, reg, name);
}

template <>
void RegPrinter::print<char, Reg8>(jit_generator &h, Reg8 reg, const char *name) {
    print_reg<char, Reg8>(h, reg, name);
}

template <>
void RegPrinter::print<unsigned char, Reg8>(jit_generator &h, Reg8 reg, const char *name) {
    print_reg<unsigned char, Reg8>(h, reg, name);
}

void RegPrinter::print_reg_fp32(const char *name, int val) {
    std::stringstream ss;
    ss << name << ": " << *reinterpret_cast<float *>(&val) << std::endl;
    std::cout << ss.str();
}

template <>
void RegPrinter::print_reg_integer<int>(const char *name, int val) {
    std::stringstream ss;
    ss << name << ": " << val << std::endl;
    std::cout << ss.str();
}

template <>
void RegPrinter::print_reg_integer<char>(const char *name, char val) {
    std::stringstream ss;
    ss << name << ": " << static_cast<int>(val) << std::endl;
    std::cout << ss.str();
}

template <>
void RegPrinter::print_reg_integer<unsigned char>(const char *name, unsigned char val) {
    std::stringstream ss;
    ss << name << ": " << static_cast<int>(val) << std::endl;
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

void RegPrinter::save_reg(jit_generator &h) {
    h.sub(h.rsp, reg_len * reg_cnt);
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(h.ptr[h.rsp + i * reg_len], Reg64(i));
    }
}

template <>
void RegPrinter::save_vmm<Xmm>(jit_generator &h) {
    h.sub(h.rsp, xmm_len * xmm_cnt);
    for (size_t i = 0; i < xmm_cnt; i++) {
        h.uni_vmovups(h.ptr[h.rsp + i * xmm_len], Xmm(i));
    }
}

template <>
void RegPrinter::save_vmm<Ymm>(jit_generator &h) {
    h.sub(h.rsp, ymm_len * ymm_cnt);
    for (size_t i = 0; i < ymm_cnt; i++) {
        h.uni_vmovups(h.ptr[h.rsp + i * ymm_len], Ymm(i));
    }
}

template <>
void RegPrinter::save_vmm<Zmm>(jit_generator &h) {
    h.sub(h.rsp, zmm_len * zmm_cnt);
    for (size_t i = 0; i < zmm_cnt; i++) {
        h.uni_vmovups(h.ptr[h.rsp + i * zmm_len], Zmm(i));
    }
}

void RegPrinter::restore_reg(jit_generator &h) {
    for (size_t i = 0; i < reg_cnt; i++) {
        h.mov(Reg64(i), h.ptr[h.rsp + i * reg_len]);
    }
    h.add(h.rsp, reg_len * reg_cnt);
}

template <>
void RegPrinter::restore_vmm<Xmm>(jit_generator &h) {
    for (size_t i = 0; i < xmm_cnt; i++) {
        h.uni_vmovups(Xmm(i), h.ptr[h.rsp + i * xmm_len]);
    }
    h.add(h.rsp, xmm_len * xmm_cnt);
}

template <>
void RegPrinter::restore_vmm<Ymm>(jit_generator &h) {
    for (size_t i = 0; i < ymm_cnt; i++) {
        h.uni_vmovups(Ymm(i), h.ptr[h.rsp + i * ymm_len]);
    }
    h.add(h.rsp, ymm_len * ymm_cnt);
}

template <>
void RegPrinter::restore_vmm<Zmm>(jit_generator &h) {
    for (size_t i = 0; i < zmm_cnt; i++) {
        h.uni_vmovups(Zmm(i), h.ptr[h.rsp + i * zmm_len]);
    }
    h.add(h.rsp, zmm_len * zmm_cnt);
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

template <typename PRC_T, typename REG_T>
void RegPrinter::print_vmm(jit_generator &h, REG_T vmm, const char *name) {
    preamble(h);

    if (name == nullptr)
        name = vmm.toString();

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

    if (name == nullptr)
        name = reg.toString();

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
