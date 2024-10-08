// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

EmitABIRegSpills::EmitABIRegSpills(jit_generator* h) : h(h), isa(get_isa()) {}

EmitABIRegSpills::~EmitABIRegSpills() {
    OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
    OPENVINO_ASSERT(rsp_status, "rsp_align or rsp_restore is missed");
}

void EmitABIRegSpills::preamble() {
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

    // Update the status
    spill_status = false;
}

void EmitABIRegSpills::postamble() {
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

    // Update the status
    spill_status = true;
}

void EmitABIRegSpills::rsp_align() {
    h->mov(h->rbx, h->rsp);
    h->and_(h->rbx, 0xf);
    h->sub(h->rsp, h->rbx);
#ifdef _WIN32
    // Allocate shadow space (home space) according to ABI
    h->sub(h->rsp, 32);
#endif

    // Update the status
    rsp_status = false;
}

void EmitABIRegSpills::rsp_restore() {
#ifdef _WIN32
    // Release shadow space (home space)
    h->add(h->rsp, 32);
#endif
    h->add(h->rsp, h->rbx);

    // Update the status
    rsp_status = true;
}

cpu_isa_t EmitABIRegSpills::get_isa() {
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
