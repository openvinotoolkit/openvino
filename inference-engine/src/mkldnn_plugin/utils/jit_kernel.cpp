// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel.hpp"
#include <stdexcept>

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace MKLDNNPlugin {

namespace {

template<typename RegType>
using registers = std::array<std::reference_wrapper<const RegType>, 16>;

template<typename RegType>
const RegType & reserveReg(jit_kernel::reg_indices & freeRegs, const registers<RegType> & regs) {
    if (freeRegs.empty())
        throw std::runtime_error("No free registers");
    const auto idx = freeRegs.back();
    freeRegs.pop_back();
    return regs[idx];
}

template<typename RegType>
void freeReg(jit_kernel::reg_indices & freeRegs, const registers<RegType> & regs, const RegType & reg) {
    const auto idx = reg.getIdx();
    // Debug:
    // auto it = std::find(freeRegs.begin(), freeRegs.end(), idx);
    // if (it != freeRegs.end())
    //     throw std::runtime_error("Some register was freed twice");
    freeRegs.emplace_back(idx);
    if (freeRegs.size() > regs.size())
        throw std::runtime_error("Some register was freed twice");
}

const registers<Reg64> & x64regs() {
    using namespace Xbyak::util;
    static const registers<Reg64> _x64regs {{
        rax, rcx, rdx, rbx,
        rsp, rbp, rsi, rdi,
        r8,  r9,  r10, r11,
        r12, r13, r14, r15,
    }};
    return _x64regs;
}

const registers<Xmm> & xmmregs() {
    static const registers<Xmm> _xmmregs {{
        util::xmm0,  util::xmm1,  util::xmm2,  util::xmm3,
        util::xmm4,  util::xmm5,  util::xmm6,  util::xmm7,
        util::xmm8,  util::xmm9,  util::xmm10, util::xmm11,
        util::xmm12, util::xmm13, util::xmm14, util::xmm15,
    }};
    return _xmmregs;
}

const registers<Ymm> & ymmregs() {
    static const registers<Ymm> _ymmregs {{
        util::ymm0,  util::ymm1,  util::ymm2,  util::ymm3,
        util::ymm4,  util::ymm5,  util::ymm6,  util::ymm7,
        util::ymm8,  util::ymm9,  util::ymm10, util::ymm11,
        util::ymm12, util::ymm13, util::ymm14, util::ymm15,
    }};
    return _ymmregs;
}

const registers<Zmm> & zmmregs() {
    static const registers<Zmm> _zmmregs {{
        util::zmm0,  util::zmm1,  util::zmm2,  util::zmm3,
        util::zmm4,  util::zmm5,  util::zmm6,  util::zmm7,
        util::zmm8,  util::zmm9,  util::zmm10, util::zmm11,
        util::zmm12, util::zmm13, util::zmm14, util::zmm15,
    }};
    return _zmmregs;
}

}   // namespace

namespace internal {

template<>
InferenceEngine::Precision type2precision<float>() {
    return InferenceEngine::Precision::FP32;
}

template<>
InferenceEngine::Precision type2precision<uint8_t>() {
    return InferenceEngine::Precision::U8;
}

cpu_isa_t get_current_isa() {
    if (mayiuse(cpu_isa_t::avx512_common))
        return cpu_isa_t::avx512_common;
    if (mayiuse(cpu_isa_t::avx2))
        return cpu_isa_t::avx2;
    return cpu_isa_t::sse41;
}

stack_frame::stack_frame(MKLDNNPlugin::jit_kernel & kernel, size_t size)
    : _kernel(kernel)
    , _size(size) {
    if (_size)
        _kernel.sub(_kernel.rsp, _size);
}

stack_frame::stack_frame(stack_frame && rhs)
    : _kernel(rhs._kernel)
    , _size(rhs._size) {
    rhs._size = 0;
}

stack_frame::~stack_frame() {
    if (_size)
        _kernel.add(_kernel.rsp, _size);
}

const Xbyak::Reg64 & stack_frame::pointer() const {
    return _kernel.rsp;
}

void stack_frame::clear() const {
    const size_t end = _size & ~(size_t)3u;

    _kernel.foreach(0, end, [&](const Reg64 & idx) {
        _kernel.mov(_kernel.qword[pointer() + idx], 0);
    }, sizeof(size_t));

    if (end < _size) {
        _kernel.foreach(end, _size, [&](const Reg64 & idx) {
            _kernel.mov(_kernel.byte[pointer() + idx], 0);
        });
    }
}

}   // namespace internal

jit_kernel::jit_kernel()
    : _load_emitter(this, internal::get_current_isa(), nullptr)
    , _store_emitter(this, internal::get_current_isa(), nullptr) {
    _free_rmmregs.reserve(16);
    _free_rmmregs.reserve(16);

    auto isRegReserved = [this](int idx) {
        return idx == param1.getIdx()           // function argument
                || idx == Operand::Code::RSP    // stack pointer
                || idx == Operand::Code::RBP;   // frame pointer
    };

    for (int reg = Operand::Code::RAX; reg <= Operand::Code::R15; ++reg) {
        if (!isRegReserved(reg))
            _free_x64regs.emplace_back(reg);
        _free_rmmregs.emplace_back(reg);
    }
}

template<>
const Reg64 & jit_kernel::reserve<Reg64>() {
    return reserveReg(_free_x64regs, x64regs());
}

template<>
void jit_kernel::free<Reg64>(const Reg64 & reg) {
    freeReg(_free_x64regs, x64regs(), reg);
}

template<>
const Xmm & jit_kernel::reserve<Xmm>() {
    return reserveReg(_free_rmmregs, xmmregs());
}

template<>
void jit_kernel::free<Xmm>(const Xmm & reg) {
    freeReg(_free_rmmregs, xmmregs(), reg);
}

template<>
const Ymm & jit_kernel::reserve<Ymm>() {
    return reserveReg(_free_rmmregs, ymmregs());
}

template<>
void jit_kernel::free<Ymm>(const Ymm & reg) {
    freeReg(_free_rmmregs, ymmregs(), reg);
}

template<>
const Zmm & jit_kernel::reserve<Zmm>() {
    return reserveReg(_free_rmmregs, zmmregs());
}

template<>
void jit_kernel::free<Zmm>(const Zmm & reg) {
    freeReg(_free_rmmregs, zmmregs(), reg);
}

void jit_kernel::postamble() {
    jit_generator::postamble();
    if (_is_load_emitter_used)
        _load_emitter.emit_data();
    if (_is_store_emitter_used)
        _store_emitter.emit_data();
}

const AddressFrame & jit_kernel::address_frame(size_t size) const {
        switch (size) {
            case 1: return byte;
            case 2: return word;
            case 4: return dword;
            case 8: return qword;
            case 16: return xword;
            case 32: return yword;
            case 64: return zword;
            default:
                break;
        }
        return ptr;
}

jit_kernel::stack_frame jit_kernel::stack(size_t size) {
    return stack_frame(*this, size);
}

void jit_kernel::uni_vpermps(const Xmm& x1, const int *mask, const Operand& op) {
    uint8_t imm8 = static_cast<uint8_t>(*mask);
    mov(x1, op);
    shufps(x1, op, imm8);
}

void jit_kernel::uni_vpermps(const Ymm& y1, const int *mask, const Operand& op) {
    auto mreg = reserve<Ymm>();
    auto mptr = reserve<Reg64>();

    mov(mptr, (size_t)mask);
    uni_vmovdqu(mreg, ptr[mptr]);
    vpermps(y1, mreg, op);

    free(mreg);
    free(mptr);
}

void jit_kernel::uni_vpermps(const Zmm& z1, const int *mask, const Operand& op) {
    auto mreg = reserve<Zmm>();
    auto mptr = reserve<Reg64>();

    mov(mptr, (size_t)mask);
    uni_vmovdqu(mreg, ptr[mptr]);
    vpermps(z1, mreg, op);

    free(mreg);
    free(mptr);
}

}   // namespace MKLDNNPlugin
