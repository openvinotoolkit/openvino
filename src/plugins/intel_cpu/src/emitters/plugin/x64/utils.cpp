// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "emitters/utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace Xbyak;
using namespace dnnl::impl::cpu::x64;

namespace {
inline snippets::Reg Xbyak2SnippetsReg(const Xbyak::Reg& xb_reg) {
    auto get_reg_type = [](const Xbyak::Reg& xb_reg) {
        switch (xb_reg.getKind()) {
        case Xbyak::Reg::REG:
            return snippets::RegType::gpr;
        case Xbyak::Reg::XMM:
        case Xbyak::Reg::YMM:
        case Xbyak::Reg::ZMM:
            return snippets::RegType::vec;
        case Xbyak::Reg::OPMASK:
            return snippets::RegType::mask;
        default:
            OPENVINO_THROW("Unhandled Xbyak reg type in conversion to snippets reg type");
        }
    };
    return {get_reg_type(xb_reg), static_cast<size_t>(xb_reg.getIdx())};
}

template <cpu_isa_t isa,
          typename std::enable_if<dnnl::impl::utils::one_of(isa, sse41, avx2, avx512_core), bool>::type = true>
struct regs_to_spill {
    static std::vector<Xbyak::Reg> get(const std::set<snippets::Reg>& live_regs) {
        std::vector<Xbyak::Reg> regs_to_spill;
        auto push_if_live = [&live_regs, &regs_to_spill](Xbyak::Reg&& reg) {
            if (live_regs.empty() || live_regs.count(Xbyak2SnippetsReg(reg)))
                regs_to_spill.emplace_back(reg);
        };
        for (int i = 0; i < 16; i++) {
            // do not spill rsp;
            if (i != 4)
                push_if_live(Reg64(i));
        }

        for (int i = 0; i < cpu_isa_traits<isa>::n_vregs; ++i)
            push_if_live(typename cpu_isa_traits<isa>::Vmm(i));

        const int num_k_mask = isa == cpu_isa_t::avx512_core ? 8 : 0;
        for (int i = 0; i < num_k_mask; ++i)
            push_if_live(Xbyak::Opmask(i));
        return regs_to_spill;
    }
};

std::vector<Xbyak::Reg> get_regs_to_spill(cpu_isa_t isa, const std::set<snippets::Reg>& live_regs) {
    switch (isa) {
    case sse41:
        return regs_to_spill<sse41>::get(live_regs);
    case avx2:
        return regs_to_spill<avx2>::get(live_regs);
    case avx512_core:
        return regs_to_spill<avx512_core>::get(live_regs);
    default:
        OPENVINO_THROW("Unhandled isa in get_regs_to_spill");
    }
}
}  // namespace

EmitABIRegSpills::EmitABIRegSpills(jit_generator* h_arg) : h(h_arg), isa(get_isa()) {}

EmitABIRegSpills::~EmitABIRegSpills() {
    OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
    OPENVINO_ASSERT(rsp_status, "rsp_align or rsp_restore is missed");
}

void EmitABIRegSpills::preamble(const std::set<snippets::Reg>& live_regs) {
    OPENVINO_ASSERT(spill_status, "Attempt to spill ABI registers twice in a row");
    // all regs to spill according to ABI
    m_regs_to_spill = get_regs_to_spill(isa, live_regs);
    for (const auto& reg : m_regs_to_spill) {
        const auto reg_bit_size = reg.getBit();
        OPENVINO_ASSERT(reg_bit_size % 8 == 0, "Unexpected reg bit size");
        m_bytes_to_spill += reg_bit_size / 8;
    }
    h->sub(h->rsp, m_bytes_to_spill);
    uint32_t byte_stack_offset = 0;
    for (const auto& reg : m_regs_to_spill) {
        Xbyak::Address addr = h->ptr[h->rsp + byte_stack_offset];
        byte_stack_offset += reg.getBit() / 8;
        switch (reg.getKind()) {
        case Xbyak::Reg::REG:
            h->mov(addr, reg);
            break;
        case Xbyak::Reg::XMM:
            h->uni_vmovups(addr, Xmm(reg.getIdx()));
            break;
        case Xbyak::Reg::YMM:
            h->uni_vmovups(addr, Ymm(reg.getIdx()));
            break;
        case Xbyak::Reg::ZMM:
            h->uni_vmovups(addr, Zmm(reg.getIdx()));
            break;
        case Xbyak::Reg::OPMASK:
            h->kmovq(addr, Opmask(reg.getIdx()));
            break;
        default:
            OPENVINO_THROW("Unhandled Xbyak reg type in conversion");
        }
    }
    // Update the status
    spill_status = false;
}

void EmitABIRegSpills::postamble() {
    OPENVINO_ASSERT(!spill_status, "Attempt to restore ABI registers that were not spilled");
    uint32_t byte_stack_offset = m_bytes_to_spill;
    for (size_t i = m_regs_to_spill.size(); i > 0; i--) {
        const auto& reg = m_regs_to_spill[i - 1];
        byte_stack_offset -= reg.getBit() / 8;
        Xbyak::Address addr = h->ptr[h->rsp + byte_stack_offset];
        switch (reg.getKind()) {
        case Xbyak::Reg::REG:
            h->mov(reg, addr);
            break;
        case Xbyak::Reg::XMM:
            h->uni_vmovups(Xmm(reg.getIdx()), addr);
            break;
        case Xbyak::Reg::YMM:
            h->uni_vmovups(Ymm(reg.getIdx()), addr);
            break;
        case Xbyak::Reg::ZMM:
            h->uni_vmovups(Zmm(reg.getIdx()), addr);
            break;
        case Xbyak::Reg::OPMASK:
            h->kmovq(Xbyak::Opmask(reg.getIdx()), addr);
            break;
        default:
            OPENVINO_THROW("Unhandled Xbyak reg type in conversion");
        }
    }
    h->add(h->rsp, m_bytes_to_spill);
    m_regs_to_spill.clear();
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
    // e.g. other emitters isa is avx512, while this emitter isa is avx2, and internal call is used. Internal call may
    // use avx512 and spoil k-reg, ZMM. do not care about platform w/ avx512_common but w/o avx512_core(knight landing),
    // which is obsoleted.
    if (mayiuse(avx512_core))
        return avx512_core;
    if (mayiuse(avx2))
        return avx2;
    if (mayiuse(sse41))
        return sse41;
    OV_CPU_JIT_EMITTER_THROW("unsupported isa");
}

}  // namespace intel_cpu
}  // namespace ov
