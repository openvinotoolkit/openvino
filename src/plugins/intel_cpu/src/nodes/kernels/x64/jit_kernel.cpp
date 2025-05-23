// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu {

namespace {

template <typename RegType>
using registers = std::array<std::reference_wrapper<const RegType>, 16>;

bool isRegAllocable(int id) {
    return id != abi_param1.getIdx()     // function argument
           && id != Operand::Code::RSP;  // stack pointer
}

template <typename RegType>
const RegType& reserveReg(jit_kernel::reg_indices& freeRegs, const registers<RegType>& regs) {
    if (freeRegs.empty()) {
        throw std::runtime_error("No free registers");
    }
    const auto idx = freeRegs.back();
    freeRegs.pop_back();
    return regs[idx];
}

template <typename RegType>
void freeReg(jit_kernel::reg_indices& freeRegs, const registers<RegType>& regs, const RegType& reg) {
    const auto idx = reg.getIdx();
    // Debug:
    // auto it = std::find(freeRegs.begin(), freeRegs.end(), idx);
    // if (it != freeRegs.end())
    //     throw std::runtime_error("Some register was freed twice");
    freeRegs.emplace_back(idx);
    if (freeRegs.size() > regs.size()) {
        OPENVINO_THROW("Some register was freed twice");
    }
}

const registers<Reg64>& x64regs() {
    using namespace Xbyak::util;
    static const registers<Reg64> _x64regs{{
        rax,
        rcx,
        rdx,
        rbx,
        rsp,
        rbp,
        rsi,
        rdi,
        r8,
        r9,
        r10,
        r11,
        r12,
        r13,
        r14,
        r15,
    }};
    return _x64regs;
}

const registers<Reg32>& x32regs() {
    using namespace Xbyak::util;
    static const registers<Reg32> _x32regs{{
        eax,
        ecx,
        edx,
        ebx,
        esp,
        ebp,
        esi,
        edi,
        r8d,
        r9d,
        r10d,
        r11d,
        r12d,
        r13d,
        r14d,
        r15d,
    }};
    return _x32regs;
}

const registers<Reg16>& x16regs() {
    using namespace Xbyak::util;
    static const registers<Reg16> _x16regs{{
        ax,
        cx,
        dx,
        bx,
        sp,
        bp,
        si,
        di,
        r8w,
        r9w,
        r10w,
        r11w,
        r12w,
        r13w,
        r14w,
        r15w,
    }};
    return _x16regs;
}

const registers<Reg8>& x8regs() {
    using namespace Xbyak::util;
    static const registers<Reg8> _x8regs{{
        al,
        cl,
        dl,
        bl,
        spl,
        bpl,
        sil,
        dil,
        r8b,
        r9b,
        r10b,
        r11b,
        r12b,
        r13b,
        r14b,
        r15b,
    }};
    return _x8regs;
}

const registers<Xmm>& xmmregs() {
    static const registers<Xmm> _xmmregs{{
        Xbyak::util::xmm0,
        Xbyak::util::xmm1,
        Xbyak::util::xmm2,
        Xbyak::util::xmm3,
        Xbyak::util::xmm4,
        Xbyak::util::xmm5,
        Xbyak::util::xmm6,
        Xbyak::util::xmm7,
        Xbyak::util::xmm8,
        Xbyak::util::xmm9,
        Xbyak::util::xmm10,
        Xbyak::util::xmm11,
        Xbyak::util::xmm12,
        Xbyak::util::xmm13,
        Xbyak::util::xmm14,
        Xbyak::util::xmm15,
    }};
    return _xmmregs;
}

const registers<Ymm>& ymmregs() {
    static const registers<Ymm> _ymmregs{{
        Xbyak::util::ymm0,
        Xbyak::util::ymm1,
        Xbyak::util::ymm2,
        Xbyak::util::ymm3,
        Xbyak::util::ymm4,
        Xbyak::util::ymm5,
        Xbyak::util::ymm6,
        Xbyak::util::ymm7,
        Xbyak::util::ymm8,
        Xbyak::util::ymm9,
        Xbyak::util::ymm10,
        Xbyak::util::ymm11,
        Xbyak::util::ymm12,
        Xbyak::util::ymm13,
        Xbyak::util::ymm14,
        Xbyak::util::ymm15,
    }};
    return _ymmregs;
}

const registers<Zmm>& zmmregs() {
    static const registers<Zmm> _zmmregs{{
        Xbyak::util::zmm0,
        Xbyak::util::zmm1,
        Xbyak::util::zmm2,
        Xbyak::util::zmm3,
        Xbyak::util::zmm4,
        Xbyak::util::zmm5,
        Xbyak::util::zmm6,
        Xbyak::util::zmm7,
        Xbyak::util::zmm8,
        Xbyak::util::zmm9,
        Xbyak::util::zmm10,
        Xbyak::util::zmm11,
        Xbyak::util::zmm12,
        Xbyak::util::zmm13,
        Xbyak::util::zmm14,
        Xbyak::util::zmm15,
    }};
    return _zmmregs;
}

}  // namespace

namespace internal {

template <>
ov::element::Type type2precision<float>() {
    return ov::element::f32;
}

template <>
ov::element::Type type2precision<int32_t>() {
    return ov::element::i32;
}

template <>
ov::element::Type type2precision<bfloat16_t>() {
    return ov::element::bf16;
}

template <>
ov::element::Type type2precision<uint8_t>() {
    return ov::element::u8;
}

template <>
ov::element::Type type2precision<int8_t>() {
    return ov::element::i8;
}

cpu_isa_t get_current_isa() {
    if (mayiuse(cpu_isa_t::avx512_core)) {
        return cpu_isa_t::avx512_core;
    }
    if (mayiuse(cpu_isa_t::avx2)) {
        return cpu_isa_t::avx2;
    }
    return cpu_isa_t::sse41;
}

stack_frame::stack_frame(ov::intel_cpu::jit_kernel& kernel, size_t size, uint32_t alignment)
    : _kernel(kernel),
      _size(size),
      _alignment(alignment) {
    if (_size || _alignment) {
        if (_size && _alignment == 1) {
            _kernel.sub(_kernel.rsp, _size);
        } else {
            auto tmp = _kernel.var<size_t>();
            tmp = _kernel.rsp;
            _kernel.sub(_kernel.rsp, sizeof(size_t) + size);    // allocate
            _kernel.and_(_kernel.rsp, ~(alignment - 1));        // align
            _kernel.mov(_kernel.ptr[_kernel.rsp + size], tmp);  // remember previous rsp
        }
    }
}

stack_frame::stack_frame(stack_frame&& rhs) noexcept
    : _kernel(rhs._kernel),
      _size(rhs._size),
      _alignment(rhs._alignment) {
    rhs._size = 0;
    rhs._alignment = 0;
}

stack_frame::~stack_frame() {
    if (_size || _alignment) {
        if (_size && _alignment == 1) {
            _kernel.add(_kernel.rsp, _size);
        } else {
            _kernel.mov(_kernel.rsp, _kernel.ptr[_kernel.rsp + _size]);
        }
    }
}

const Xbyak::Reg64& stack_frame::pointer() const {
    return _kernel.rsp;
}

void stack_frame::clear() const {
    const size_t end = _size & ~static_cast<size_t>(7u);

    _kernel.foreach (
        0,
        end,
        [&](const Reg64& idx) {
            _kernel.mov(_kernel.qword[pointer() + idx], 0);
        },
        sizeof(size_t));

    if (end < _size) {
        _kernel.foreach (end, _size, [&](const Reg64& idx) {
            _kernel.mov(_kernel.byte[pointer() + idx], 0);
        });
    }
}

const void* consts_table::store(const void* data, size_t size) {
    if (size > chunk_size) {
        throw std::runtime_error("Data size is too large");
    }
    const size_t capacity = _chunks.size() * chunk_size;
    if (size > capacity - _size) {
        _size = _chunks.size() * chunk_size;
        _chunks.emplace_back();
    }
    auto& dst = _chunks.back();
    const size_t offset = _size % chunk_size;
    memcpy(&dst[offset], data, size);
    _size += size;
    return &dst[offset];
}

}  // namespace internal

jit_kernel::jit_kernel(const char* name) : jit_generator(name) {
    _free_rmmregs.reserve(16);
    _free_rmmregs.reserve(16);

    for (int reg = Operand::Code::RAX; reg <= Operand::Code::R15; ++reg) {
        if (isRegAllocable(reg)) {
            _free_x64regs.emplace_back(reg);
        }
        _free_rmmregs.emplace_back(reg);
    }
}

template <>
const Reg64& jit_kernel::reserve<Reg64>() {
    return reserveReg(_free_x64regs, x64regs());
}

template <>
const Reg32& jit_kernel::reserve<Reg32>() {
    return reserveReg(_free_x64regs, x32regs());
}

template <>
const Reg16& jit_kernel::reserve<Reg16>() {
    return reserveReg(_free_x64regs, x16regs());
}

template <>
const Reg8& jit_kernel::reserve<Reg8>() {
    return reserveReg(_free_x64regs, x8regs());
}

template <>
void jit_kernel::free<Reg64>(const Reg64& reg) {
    freeReg(_free_x64regs, x64regs(), reg);
}

template <>
void jit_kernel::free<Reg32>(const Reg32& reg) {
    freeReg(_free_x64regs, x32regs(), reg);
}

template <>
void jit_kernel::free<Reg16>(const Reg16& reg) {
    freeReg(_free_x64regs, x16regs(), reg);
}

template <>
void jit_kernel::free<Reg8>(const Reg8& reg) {
    freeReg(_free_x64regs, x8regs(), reg);
}

template <>
const Xmm& jit_kernel::reserve<Xmm>() {
    return reserveReg(_free_rmmregs, xmmregs());
}

template <>
void jit_kernel::free<Xmm>(const Xmm& reg) {
    freeReg(_free_rmmregs, xmmregs(), reg);
}

template <>
const Ymm& jit_kernel::reserve<Ymm>() {
    return reserveReg(_free_rmmregs, ymmregs());
}

template <>
void jit_kernel::free<Ymm>(const Ymm& reg) {
    freeReg(_free_rmmregs, ymmregs(), reg);
}

template <>
const Zmm& jit_kernel::reserve<Zmm>() {
    return reserveReg(_free_rmmregs, zmmregs());
}

template <>
void jit_kernel::free<Zmm>(const Zmm& reg) {
    freeReg(_free_rmmregs, zmmregs(), reg);
}

void jit_kernel::postamble() {
    jit_generator::postamble();
    for (const auto& emitter : _emitters) {
        if (emitter.second) {
            emitter.second->emit_data();
        }
    }
}

const AddressFrame& jit_kernel::address_frame(size_t size) const {
    switch (size) {
    case 1:
        return byte;
    case 2:
        return word;
    case 4:
        return dword;
    case 8:
        return qword;
    case 16:
        return xword;
    case 32:
        return yword;
    case 64:
        return zword;
    default:
        break;
    }
    return ptr;
}

const jit_kernel::reg_indices& jit_kernel::free_x64regs() const {
    return _free_x64regs;
}

const jit_kernel::reg_indices& jit_kernel::free_rmmregs() const {
    return _free_rmmregs;
}

jit_kernel::stack_frame jit_kernel::stack(size_t size, uint32_t alignment) {
    return {*this, size, alignment};
}

void jit_kernel::uni_vpermps(const Xmm& x1, const uint8_t mask[4], const Operand& op) {
    uint8_t imm8 = 0;
    for (size_t i = 0; i < 4; ++i) {
        imm8 |= mask[i] << (i * 2);
    }
    if (op != x1) {
        movdqu(x1, op);
    }
    shufps(x1, op, imm8);
}

void jit_kernel::uni_vpermps(const Ymm& y1, const uint8_t mask[8], const Operand& op) {
    int data[8];
    for (size_t i = 0; i < 8; ++i) {
        data[i] = mask[i];
    }
    auto mreg = var<int[8]>();
    mreg = data;
    vpermps(y1, mreg, op);
}

void jit_kernel::uni_vpermps(const Zmm& z1, const uint8_t mask[16], const Operand& op) {
    int data[16];
    for (size_t i = 0; i < 16; ++i) {
        data[i] = mask[i];
    }
    auto mreg = var<int[16]>();
    mreg = data;
    vpermps(z1, mreg, op);
}

void jit_kernel::uni_vblendps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, uint16_t mask) {
    blendps(x1, x2, mask);
}

void jit_kernel::uni_vblendps(const Xbyak::Ymm& y1, const Xbyak::Ymm& y2, uint16_t mask) {
    vblendps(y1, y1, y2, static_cast<uint8_t>(mask));
}

void jit_kernel::uni_vblendps(const Xbyak::Zmm& z1, const Xbyak::Zmm& z2, uint16_t mask) {
    auto reg = var<uint32_t>();
    mov(reg, mask);
    kmovw(k1, reg);
    vblendmps(z1 | k1, z1, z2);
}

}  // namespace ov::intel_cpu
