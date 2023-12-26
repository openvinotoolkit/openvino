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

#ifdef SNIPPETS_DEBUG_CAPS
std::string get_type_name(const jit_emitter* emitter);
std::string get_type_name(const jit_emitter* emitter) {
    std::string name = typeid(*emitter).name();
#ifndef _WIN32
    int status;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
            abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
            std::free);
    name = demangled_name.get();
#endif
    return name;
}

void print_segfault_detector_result(jit_uni_segfault_detector_emitter* detector_emitter) {
    auto print_memory_emitter_info = [&](MemoryEmitter *memory_emitter) {
        std::cerr << "detailed emitter info is, src precision:" << memory_emitter->src_prc << ", dst precision:" << memory_emitter->dst_prc
            << ", load/store element number:" << memory_emitter->count
            << ", byte offset" << memory_emitter->byte_offset << std::endl;
        // more memory address info tracked in detector_emitter.
        std::cerr << "start_address:" << detector_emitter->start_address
            << ", current_address:" << detector_emitter->current_address
            << ", iteration:" << detector_emitter->iteration << "\n";
    };
    auto print_brgemm_emitter_info = [&](BrgemmEmitter* brgemm_emitter) {
        std::cerr << "detailed emitter info is, m_brgCtx.M:" << brgemm_emitter->m_brgCtx.M
            << " m_brgCtx.K:" << brgemm_emitter->m_brgCtx.K
            << " m_brgCtx.N:" << brgemm_emitter->m_brgCtx.N
            << " m_brgCtx.LDA:" << brgemm_emitter->m_brgCtx.LDA
            << " m_brgCtx.LDB:" << brgemm_emitter->m_brgCtx.LDB
            << " m_brgCtx.LDC:" << brgemm_emitter->m_brgCtx.LDC
            << " m_brgCtx.dt_in0:" << brgemm_emitter->m_brgCtx.dt_in0
            << " m_brgCtx.dt_in1:" << brgemm_emitter->m_brgCtx.dt_in1
            << " m_brgCtx.palette:" << brgemm_emitter->m_brgCtx.palette
            << " m_brgCtx.is_with_amx:" << brgemm_emitter->m_brgCtx.is_with_amx
            << " m_brgCtx.is_with_comp:" << brgemm_emitter->m_brgCtx.is_with_comp
            << " m_brgCtx.beta:" << brgemm_emitter->m_brgCtx.beta
            << " m_load_offset_a:" << brgemm_emitter->m_load_offset_a
            << " m_load_offset_b:" << brgemm_emitter->m_load_offset_b
            << " m_load_offset_scratch:" << brgemm_emitter->m_load_offset_scratch
            << " m_store_offset_c:" << brgemm_emitter->m_store_offset_c
            << " m_with_scratch:" << brgemm_emitter->m_with_scratch
            << " m_with_comp:" << brgemm_emitter->m_with_comp << "\n";
    };
    auto print_brgemm_copy_emitter_info = [&](BrgemmCopyBEmitter* brgemm_copy_emitter) {
        std::cerr << "detailed emitter info is, m_LDB:" << brgemm_copy_emitter->m_LDB
            << " m_K:" << brgemm_copy_emitter->m_K
            << " m_K_blk:" << brgemm_copy_emitter->m_K_blk
            << " m_K_tail:" << brgemm_copy_emitter->m_K_tail
            << " m_N:" << brgemm_copy_emitter->m_N
            << " m_N_blk:" << brgemm_copy_emitter->m_N_blk
            << " m_N_tail:" << brgemm_copy_emitter->m_N_tail
            << " m_brgemm_prc_in0:" << brgemm_copy_emitter->m_brgemm_prc_in0
            << " m_brgemm_prc_in1:" << brgemm_copy_emitter->m_brgemm_prc_in1
            << " m_brgemmVNNIFactor:" << brgemm_copy_emitter->m_brgemmVNNIFactor
            << " m_with_comp:" << brgemm_copy_emitter->m_with_comp
            << " m_in_offset:" << brgemm_copy_emitter->m_in_offset
            << " m_out_offset:" << brgemm_copy_emitter->m_out_offset
            << " m_comp_offset:" << brgemm_copy_emitter->m_comp_offset << "\n";
    };
    std::cerr << "Node name:" << detector_emitter->m_target_node_name << std::endl;
    auto target_emitter = detector_emitter->m_target_emitter;
    std::cerr << "Emitter type name:" << get_type_name(target_emitter) << std::endl;
    if (auto *memory_emitter = dynamic_cast<MemoryEmitter *>(target_emitter)) {
        print_memory_emitter_info(memory_emitter);
    } else if (auto *brgemm_emitter = dynamic_cast<BrgemmEmitter *>(target_emitter)) {
        print_brgemm_emitter_info(brgemm_emitter);
    } else if (auto *brgemm_copy_emitter = dynamic_cast<BrgemmCopyBEmitter *>(target_emitter)) {
        print_brgemm_copy_emitter_info(brgemm_copy_emitter);
    }
}

#endif

}   // namespace intel_cpu
}   // namespace ov
