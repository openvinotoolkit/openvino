// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel.hpp"

#include <xbyak/xbyak.h>

#include <array>
#include <common/bfloat16.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu {

namespace {

template <typename RegType>
using registers = std::array<std::reference_wrapper<const RegType>, 16>;

bool ir_trace_enabled() {
    return std::getenv("OV_JIT_IR_TRACE") != nullptr;
}

void ir_trace(const std::string& msg) {
    if (!ir_trace_enabled()) {
        return;
    }
    std::cout << "[jit_kernel_ir] " << msg << "\n";
}

bool isRegAllocable(int id) {
    return id != abi_param1.getIdx()     // function argument
           && id != Operand::Code::RSP   // stack pointer
           && id != Operand::Code::RBP;  // frame pointer (used by preamble/postamble)
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
    OPENVINO_ASSERT(freeRegs.size() <= regs.size(), "Some register was freed twice");
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
    const size_t end = _size & ~static_cast<size_t>(7U);

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

jit_kernel::jit_kernel(const char* name) : jit_generator_t(name) {
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
    jit_generator_t::postamble();
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

void jit_kernel::uni_vblendps(const Xbyak::Xmm& dst,
                              const Xbyak::Xmm& src1,
                              const Xbyak::Xmm& src2,
                              uint16_t mask) {
    vblendps(dst, src1, src2, static_cast<uint8_t>(mask));
}

void jit_kernel::uni_vblendps(const Xbyak::Ymm& dst,
                              const Xbyak::Ymm& src1,
                              const Xbyak::Ymm& src2,
                              uint16_t mask) {
    vblendps(dst, src1, src2, static_cast<uint8_t>(mask));
}

void jit_kernel::uni_vblendps(const Xbyak::Zmm& dst,
                              const Xbyak::Zmm& src1,
                              const Xbyak::Zmm& src2,
                              uint16_t mask) {
    auto reg = var<uint32_t>();
    mov(reg, mask);
    kmovw(k1, reg);
    vblendmps(dst | k1, src1, src2);
}

// ── IR mode ────────────────────────────────────────────────────────────

void jit_kernel::ir_use(std::vector<jit_kernel_ir::value_id> reads,
                        jit_kernel_ir::EmitFn emit,
                        const char* name,
                        jit_kernel_ir::OpKind kind) {
    _ir->use(std::move(reads), std::move(emit), name, kind);
}

jit_kernel_ir::value_id jit_kernel::ir_def_gpr(std::vector<jit_kernel_ir::value_id> reads,
                                                 jit_kernel_ir::EmitFn emit,
                                                 const char* name) {
    return _ir->def(std::move(reads), std::move(emit), name, jit_kernel_ir::RegisterClass::GPR);
}

// GPR arithmetic helpers — LLVM-style: def_tied + TwoAddressPass.
// The tied operand constraint lets the allocator coalesce the copy
// when the source dies, producing mov-free code like hand-written asm.

jit_kernel::variable<size_t> jit_kernel::ir_shr(const variable<size_t>& src, int shift) {
    return variable<size_t>(*this, _ir->def_tied({src.vid()}, 0,
        [this, shift](const jit_kernel_ir::EmitContext& ctx) {
            shr(Xbyak::Reg64(ctx.def->idx), shift);
        }, "shr", jit_kernel_ir::RegisterClass::GPR));
}

jit_kernel::variable<size_t> jit_kernel::ir_and(const variable<size_t>& src, size_t mask) {
    return variable<size_t>(*this, _ir->def_tied({src.vid()}, 0,
        [this, mask](const jit_kernel_ir::EmitContext& ctx) {
            and_(Xbyak::Reg64(ctx.def->idx), mask);
        }, "and", jit_kernel_ir::RegisterClass::GPR));
}

jit_kernel::variable<size_t> jit_kernel::ir_add(const variable<size_t>& src, size_t val) {
    return variable<size_t>(*this, _ir->def_tied({src.vid()}, 0,
        [this, val](const jit_kernel_ir::EmitContext& ctx) {
            add(Xbyak::Reg64(ctx.def->idx), val);
        }, "add", jit_kernel_ir::RegisterClass::GPR));
}

jit_kernel::variable<size_t> jit_kernel::ir_imul(const variable<size_t>& src, size_t val) {
    // imul is 3-operand (non-destructive) — no tied constraint needed.
    return variable<size_t>(*this, _ir->def({src.vid()},
        [this, val](const jit_kernel_ir::EmitContext& ctx) {
            imul(Xbyak::Reg64(ctx.def->idx), Xbyak::Reg64(ctx.reads[0].idx),
                 static_cast<int>(val));
        }, "imul", jit_kernel_ir::RegisterClass::GPR));
}

jit_kernel_ir::value_id jit_kernel::ir_alloca(size_t size, size_t alignment) {
    auto alloca_idx = _alloca_requests.size();
    _alloca_requests.push_back({size, alignment, 0});

    return _ir->def({}, [this, alloca_idx](const jit_kernel_ir::EmitContext& ctx) {
        lea(Xbyak::Reg64(ctx.def->idx),
            ptr[rsp + _alloca_requests[alloca_idx].offset]);
    }, "alloca", jit_kernel_ir::RegisterClass::GPR);
}

void jit_kernel::begin_ir() {
    OPENVINO_ASSERT(!_ir, "begin_ir() called while already in IR mode");
    _ir = std::make_unique<jit_kernel_ir::IR>();
}

void jit_kernel::end_ir() {
    if (!_ir) return;

    // Unroll strategy from env var.
    static const char* unroll_env = std::getenv("OV_JIT_IR_UNROLL");
    auto unroll_strategy = jit_kernel_ir::UnrollStrategy::none;
    if (unroll_env) {
        std::string val(unroll_env);
        if (val == "heuristic") unroll_strategy = jit_kernel_ir::UnrollStrategy::heuristic;
        else if (val == "feedback") unroll_strategy = jit_kernel_ir::UnrollStrategy::feedback;
    }

    // Build pass pipeline.
    auto pm = jit_kernel_ir::build_default_pipeline(unroll_strategy);

    // Set up pass context with dual register pools.
    jit_kernel_ir::PassContext ctx;
    ctx.vec_pool_size = static_cast<std::uint32_t>(_free_rmmregs.size());
    // GPR pool: actual register indices (non-contiguous after arg() reserves).
    // Mirrors LLVM's AllocationOrder — the allocator picks from this list.
    ctx.gpr_pool_indices.assign(_free_x64regs.begin(), _free_x64regs.end());
    ctx.dump = std::getenv("OV_JIT_IR_DUMP") != nullptr;
    ctx.trace = std::getenv("OV_JIT_IR_TRACE") != nullptr;

    // Lowering callback — walks the IR tree and emits xbyak instructions.
    ctx.lower_fn = [this](const jit_kernel_ir::IR& ir,
                          const jit_kernel_ir::Assignment& assignment) {
        std::function<void(const std::list<jit_kernel_ir::Op>&)> lower;
        lower = [&](const std::list<jit_kernel_ir::Op>& ops) {
            for (const auto& op : ops) {
                if (op.body) {
                    ir_trace(std::string("lower ") + (op.is_loop ? "loop" : "region") + " enter");
                    // Resolve reads for region ops (loop header idx/end etc.)
                    std::vector<jit_kernel_ir::PhysReg> read_regs;
                    read_regs.reserve(op.reads.size());
                    for (auto vid : op.reads) {
                        read_regs.push_back(assignment.reg.at(vid));
                    }
                    const jit_kernel_ir::EmitContext ectx{std::nullopt, read_regs};
                    op.emit(ectx);
                    lower(op.body->ops());
                    ir_trace(std::string("lower ") + (op.is_loop ? "loop" : "region") + " exit");
                } else {
                    std::vector<jit_kernel_ir::PhysReg> read_regs;
                    read_regs.reserve(op.reads.size());
                    for (auto vid : op.reads) {
                        read_regs.push_back(assignment.reg.at(vid));
                    }
                    std::optional<jit_kernel_ir::PhysReg> def_reg;
                    if (op.def != jit_kernel_ir::invalid_value) {
                        def_reg = assignment.reg.at(op.def);
                    }
                    if (ir_trace_enabled()) {
                        std::ostringstream os;
                        os << "lower op";
                        if (op.name[0]) os << " name=" << op.name;
                        if (def_reg) os << " def=%" << op.def << "->p" << def_reg->idx;
                        else os << " def=-";
                        os << " reads=[";
                        for (std::size_t i = 0; i < op.reads.size(); ++i) {
                            if (i) os << ", ";
                            os << "%" << op.reads[i] << "->p" << read_regs[i].idx;
                        }
                        os << "]";
                        ir_trace(os.str());
                    }
                    const jit_kernel_ir::EmitContext ectx{def_reg, read_regs};
                    // TwoAddressPass COPY: dispatch by register class.
                    // LLVM-style: X86InstrInfo::copyPhysReg checks class.
                    if (op.is_copy && def_reg && !read_regs.empty()
                        && def_reg->idx != read_regs[0].idx) {
                        if (op.def_rc == jit_kernel_ir::RegisterClass::GPR) {
                            // GPR copy: mov reg64, reg64
                            mov(Xbyak::Reg64(def_reg->idx),
                                Xbyak::Reg64(read_regs[0].idx));
                        } else {
                            // Vec copy: vmovups
                            using namespace dnnl::impl::cpu::x64;
                            if (mayiuse(avx512_core)) {
                                uni_vmovups(Xbyak::Zmm(def_reg->idx),
                                            Xbyak::Zmm(read_regs[0].idx));
                            } else {
                                uni_vmovups(Xbyak::Ymm(def_reg->idx),
                                            Xbyak::Ymm(read_regs[0].idx));
                            }
                        }
                    } else {
                        op.emit(ectx);
                    }
                }
            }
        };
        lower(ir.ops());
    };

    // Compute alloca offsets. One contiguous stack frame for all ir_alloca calls.
    size_t total_alloca = 0;
    for (auto& req : _alloca_requests) {
        total_alloca = (total_alloca + req.alignment - 1) & ~(req.alignment - 1);
        req.offset = total_alloca;
        total_alloca += req.size;
    }
    // Align total to 16 bytes (ABI requirement for stack alignment).
    total_alloca = (total_alloca + 15) & ~static_cast<size_t>(15);

    if (total_alloca > 0) {
        sub(rsp, total_alloca);
    }

    // Run the pipeline.
    pm.run(*_ir, ctx);

    if (total_alloca > 0) {
        add(rsp, total_alloca);
    }

    _alloca_requests.clear();
    _ir.reset();
}

}  // namespace ov::intel_cpu
