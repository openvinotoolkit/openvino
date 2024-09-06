// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include <iostream>
#include <sstream>

#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

#ifdef CPU_DEBUG_CAPS

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

template <typename T>
void RegPrinter::print_reg_prc(const char *name, const char *ori_name, T *ptr) {
    std::stringstream ss;
    if (name) ss << name << " | ";
    ss << ori_name << ": ";
    if (std::is_floating_point<T>::value) {
        ss << *ptr;
    } else {
        if (std::is_signed<T>::value) {
            ss << static_cast<int64_t>(*ptr);
        } else {
            ss << static_cast<uint64_t>(*ptr);
        }
    }
    ss << std::endl;
    std::cout << ss.str();
}

template <typename PRC_T, size_t vlen>
void RegPrinter::print_vmm_prc(const char *name, const char *ori_name, PRC_T *ptr) {
    std::stringstream ss;
    if (name) ss << name << " | ";
    ss << ori_name << ": {" << ptr[0];
    for (size_t i = 1; i < vlen / sizeof(float); i++) {
        ss << ", " << ptr[i];
    }
    ss << "}" << std::endl;
    std::cout << ss.str();
}
template void RegPrinter::print_vmm_prc<float, 16>(const char *name, const char *ori_name, float *ptr);
template void RegPrinter::print_vmm_prc<float, 32>(const char *name, const char *ori_name, float *ptr);
template void RegPrinter::print_vmm_prc<float, 64>(const char *name, const char *ori_name, float *ptr);
template void RegPrinter::print_vmm_prc<int, 16>(const char *name, const char *ori_name, int *ptr);
template void RegPrinter::print_vmm_prc<int, 32>(const char *name, const char *ori_name, int *ptr);
template void RegPrinter::print_vmm_prc<int, 64>(const char *name, const char *ori_name, int *ptr);

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

// ABI requires 16-bype stack alignment before a call
void RegPrinter::align_rsp(jit_generator &h) {
    constexpr int alignment = 16;
    h.mov(h.r15, h.rsp);
    h.and_(h.rsp, ~(alignment - 1));
}

void RegPrinter::restore_rsp(jit_generator &h) {
    h.mov(h.rsp, h.r15);
}

template <typename PRC_T, typename REG_T>
void RegPrinter::print_vmm(jit_generator &h, REG_T vmm, const char *name) {
    preamble(h);

    h.push(h.rax);
    h.push(abi_param1);
    h.push(abi_param2);
    h.push(abi_param3);
    {
        const int vlen = vmm.isZMM() ? 64 : (vmm.isYMM() ? 32 : 16);
        h.sub(h.rsp, vlen);
        h.uni_vmovups(h.ptr[h.rsp], vmm);

        h.mov(abi_param3, h.rsp);
        h.mov(abi_param2, reinterpret_cast<size_t>(vmm.toString()));
        h.mov(abi_param1, reinterpret_cast<size_t>(name));
        if (vmm.isZMM()) {
            auto p = &print_vmm_prc<PRC_T, 64>;
            h.mov(h.rax, reinterpret_cast<size_t>(p));
        } else if (vmm.isYMM()) {
            auto p = &print_vmm_prc<PRC_T, 32>;
            h.mov(h.rax, reinterpret_cast<size_t>(p));
        } else {
            auto p = &print_vmm_prc<PRC_T, 16>;
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
void RegPrinter::print_reg(jit_generator &h, REG_T reg, const char *name) {
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
        auto p = &print_reg_prc<PRC_T>;
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

#endif

JitSafeInternalCall::JitSafeInternalCall(jit_generator* h) : h(h), isa(get_isa()) {
    // gprs
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

    h->sub(h->rsp, n_gprs_to_save * gpr_size);
    for (size_t i = 0; i < n_gprs_to_save; ++i)
        h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

    if (isa == avx512_core) {
        h->sub(h->rsp, k_mask_num * k_mask_size);
        for (size_t i = 0; i < k_mask_num; ++i) {
            h->kmovq(h->ptr[h->rsp + i * k_mask_size], Xbyak::Opmask(static_cast<int>(i)));
        }
    }

    h->sub(h->rsp, get_max_vecs_count() * get_vec_length());
    for (size_t i = 0; i < get_max_vecs_count(); ++i) {
        const auto addr = h->ptr[h->rsp + i * get_vec_length()];
        if (isa == sse41) {
            h->uni_vmovups(addr, Xmm(i));
        } else if (isa == avx2) {
            h->uni_vmovups(addr, Ymm(i));
        } else {
            h->uni_vmovups(addr, Zmm(i));
        }
    }
}

JitSafeInternalCall::~JitSafeInternalCall() {
    // restore vector registers
    for (int i = static_cast<int>(get_max_vecs_count()) - 1; i >= 0; --i) {
        const auto addr = h->ptr[h->rsp + i * get_vec_length()];
        if (isa == sse41) {
            h->uni_vmovups(Xmm(i), addr);
        } else if (isa == avx2) {
            h->uni_vmovups(Ymm(i), addr);
        } else {
            h->uni_vmovups(Zmm(i), addr);
        }
    }
    h->add(h->rsp, (get_max_vecs_count()) * get_vec_length());

    // restore k reg
    if (isa == avx512_core) {
        for (int i = k_mask_num - 1; i >= 0; --i) {
            h->kmovq(Xbyak::Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
        }
        h->add(h->rsp, k_mask_num * k_mask_size);
    }

    // restore gpr registers
    Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                     h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
    size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);
    for (int i = n_gprs_to_save - 1; i >= 0; --i)
        h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
    h->add(h->rsp, n_gprs_to_save * gpr_size);
}

void JitSafeInternalCall::call(Xbyak::Reg64 reg) const {
    rsp_align();
    h->call(reg);
    rsp_restore();
}

void JitSafeInternalCall::rsp_align() const {
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);
#ifdef _WIN32
    // Allocate shadow space (home space) according to ABI
    h->sub(h->rsp, 32);
#endif
}

void JitSafeInternalCall::rsp_restore() const {
#ifdef _WIN32
    // Release shadow space (home space)
    h->add(h->rsp, 32);
#endif
    h->add(h->rsp, h->rbx);
}

cpu_isa_t JitSafeInternalCall::get_isa() {
    // need preserve based on cpu capability, instead of host isa.
    // in case there are possibilty that different isa emitters exist in one kernel from perf standpoint in the future.
    // e.g. other emitters isa is avx512, while this emitter isa is avx2, and internal call is used. Internal call may use avx512 and spoil k-reg, ZMM.
    // do not care about platform w/ avx512_common but w/o avx512_core(knight landing), which is obsoleted.
    if (mayiuse(avx512_core)) return avx512_core;
    if (mayiuse(avx2)) return avx2;
    if (mayiuse(sse41)) return sse41;
    OV_CPU_JIT_EMITTER_THROW("unsupported isa");
}

}   // namespace intel_cpu
}   // namespace ov
