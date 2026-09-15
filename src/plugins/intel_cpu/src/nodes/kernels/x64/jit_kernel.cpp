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
#include <cstdlib>
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

// Vector register file sizes. AVX-512 has 32 architectural vector
// registers; everything older has 16.
constexpr std::size_t vec_reg_count_legacy = 16;
constexpr std::size_t vec_reg_count_avx512 = 32;

// Vector register table indexed by physical register number, always the
// full 32 entries — which entries may be used is decided by the pool, not
// by the table.
template <typename VecType>
const std::vector<VecType>& vec_table() {
    static const std::vector<VecType> table = [] {
        std::vector<VecType> regs;
        regs.reserve(vec_reg_count_avx512);
        for (std::size_t i = 0; i < vec_reg_count_avx512; ++i) {
            regs.emplace_back(static_cast<int>(i));
        }
        return regs;
    }();
    return table;
}

template <typename RegType, typename Table>
const RegType& reserveReg(jit_kernel::reg_indices& freeRegs, const Table& regs) {
    if (freeRegs.empty()) {
        throw std::runtime_error("No free registers");
    }
    const auto idx = freeRegs.back();
    freeRegs.pop_back();
    return regs[idx];
}

template <typename RegType, typename Table>
void freeReg(jit_kernel::reg_indices& freeRegs, const Table& regs, const RegType& reg) {
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

const std::vector<Xmm>& xmmregs() {
    return vec_table<Xmm>();
}

const std::vector<Ymm>& ymmregs() {
    return vec_table<Ymm>();
}

const std::vector<Zmm>& zmmregs() {
    return vec_table<Zmm>();
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

// How many full iterations foreach_vec() may emit straight-line for a
// constant trip count. Four is one cache line of bodies for the kernels we
// have; OV_JIT_IR_PEEL overrides it, and 0 forces the rolled loop, which is
// how the two shapes get compared without a rebuild.
size_t jit_kernel::default_peel_limit() {
    static const size_t value = [] {
        const char* env = std::getenv("OV_JIT_IR_PEEL");
        return env != nullptr ? std::strtoul(env, nullptr, 10) : 4U;
    }();
    return value;
}

jit_kernel::jit_kernel(const char* name) : jit_generator_t(name) {
    for (int reg = Operand::Code::RAX; reg <= Operand::Code::R15; ++reg) {
        if (isRegAllocable(reg)) {
            _free_x64regs.emplace_back(reg);
        }
    }

    // The vector file is sized independently of the GPR file — it used to be
    // filled from the GPR index range, which was a coincidence of both being
    // 16 wide. Only the low 16 registers go in the eager pool: xmm16..31 and
    // ymm16..31 exist on AVX-512 but are EVEX-only, so VEX/SSE-encoded
    // instructions (which eager kernels emit freely) cannot reference them.
    // IR mode can opt into the upper half via set_vec_width(512) — see
    // end_ir().
    _free_rmmregs.reserve(vec_reg_count_legacy);
    for (size_t reg = 0; reg < vec_reg_count_legacy; ++reg) {
        _free_rmmregs.emplace_back(static_cast<size_t>(reg));
    }
}

template <>
const Reg64& jit_kernel::reserve<Reg64>() {
    return reserveReg<Reg64>(_free_x64regs, x64regs());
}

template <>
const Reg32& jit_kernel::reserve<Reg32>() {
    return reserveReg<Reg32>(_free_x64regs, x32regs());
}

template <>
const Reg16& jit_kernel::reserve<Reg16>() {
    return reserveReg<Reg16>(_free_x64regs, x16regs());
}

template <>
const Reg8& jit_kernel::reserve<Reg8>() {
    return reserveReg<Reg8>(_free_x64regs, x8regs());
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
    return reserveReg<Xmm>(_free_rmmregs, xmmregs());
}

template <>
void jit_kernel::free<Xmm>(const Xmm& reg) {
    freeReg(_free_rmmregs, xmmregs(), reg);
}

template <>
const Ymm& jit_kernel::reserve<Ymm>() {
    return reserveReg<Ymm>(_free_rmmregs, ymmregs());
}

template <>
void jit_kernel::free<Ymm>(const Ymm& reg) {
    freeReg(_free_rmmregs, ymmregs(), reg);
}

template <>
const Zmm& jit_kernel::reserve<Zmm>() {
    return reserveReg<Zmm>(_free_rmmregs, zmmregs());
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
                        const char* name) {
    _ir->use(std::move(reads), std::move(emit), name);
}

jit_kernel_ir::value_id jit_kernel::ir_def_gpr(std::vector<jit_kernel_ir::value_id> reads,
                                                 jit_kernel_ir::EmitFn emit,
                                                 const char* name) {
    return _ir->def(std::move(reads), std::move(emit), name, jit_kernel_ir::RegisterClass::GPR);
}

jit_kernel_ir::value_id jit_kernel::ir_def_mask(std::vector<jit_kernel_ir::value_id> reads,
                                                  jit_kernel_ir::EmitFn emit,
                                                  const char* name) {
    return _ir->def(std::move(reads), std::move(emit), name, jit_kernel_ir::RegisterClass::Mask);
}

// GPR arithmetic helpers — LLVM-style: def_tied + TwoAddressPass.
// The tied operand constraint lets the allocator coalesce the copy
// when the source dies, producing mov-free code like hand-written asm.

jit_kernel::variable<size_t> jit_kernel::ir_gpr_imm(size_t value) {
    return variable<size_t>(*this, _ir->def({},
        [this, value](const jit_kernel_ir::EmitContext& ctx) {
            mov(Xbyak::Reg64(ctx.def->idx), value);
        }, "imm", jit_kernel_ir::RegisterClass::GPR));
}

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

    // Build pass pipeline.
    auto pm = jit_kernel_ir::build_default_pipeline();

    // Set up pass context with dual register pools.
    jit_kernel_ir::PassContext ctx;
    // Allocation orders: the physical registers still free at this point.
    // Both lists are explicit — eager reservations (arg(), reserve<>())
    // remove registers from them, so the allocator never hands out a
    // register the kernel is already holding.
    ctx.vec_pool_indices.assign(_free_rmmregs.begin(), _free_rmmregs.end());

    // zmm16..zmm31 are available only to kernels that work exclusively in
    // 512-bit vectors: those registers require EVEX encoding, so a kernel
    // emitting any VEX-only instruction (vblendps, shufps, vperm2i128, ...)
    // on a narrower value must not be given one. Kernels declare their
    // width with set_vec_width(); the default keeps the legacy 16.
    if (_vec_width_bits == 512 && mayiuse(cpu_isa_t::avx512_core)) {
        for (size_t reg = vec_reg_count_legacy; reg < vec_reg_count_avx512; ++reg) {
            ctx.vec_pool_indices.push_back(static_cast<std::uint32_t>(reg));
        }
    }

    // Predicate registers come from the target: it owns the constraint
    // (x86 cannot use k0 as a write-mask). ISAs without a predicate file
    // return an empty order, so any attempt to allocate a mask fails
    // loudly instead of picking a register that does not exist.
    const auto& predicates = target().predicate_pool();
    ctx.mask_pool_indices.assign(predicates.begin(), predicates.end());
    ctx.gpr_pool_indices.assign(_free_x64regs.begin(), _free_x64regs.end());
    ctx.dump = std::getenv("OV_JIT_IR_DUMP") != nullptr;
    // A/B switch for the folding pass, same purpose as LLVM's
    // -disable-peephole: measure what the transform is worth.
    ctx.disable_memory_folding = std::getenv("OV_JIT_IR_NO_FOLD") != nullptr;
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
                    const jit_kernel_ir::EmitContext ectx{std::nullopt, read_regs, std::nullopt};
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
                    // A folded operand resolves to (base register, displacement);
                    // the op's emit closure builds the address form.
                    std::optional<jit_kernel_ir::FoldedMem> folded;
                    if (op.folded_read >= 0 &&
                        op.folded_read < static_cast<int>(read_regs.size())) {
                        folded = jit_kernel_ir::FoldedMem{
                            read_regs[static_cast<std::size_t>(op.folded_read)], op.mem_offset,
                            op.folded_read};
                    }
                    const jit_kernel_ir::EmitContext ectx{def_reg, read_regs, folded};
                    // TwoAddressPass COPY: dispatch by register class.
                    // LLVM-style: X86InstrInfo::copyPhysReg checks class.
                    if (op.is_copy && def_reg && !read_regs.empty()
                        && def_reg->idx != read_regs[0].idx) {
                        if (op.def_rc == jit_kernel_ir::RegisterClass::GPR) {
                            // GPR copy: mov reg64, reg64
                            mov(Xbyak::Reg64(def_reg->idx),
                                Xbyak::Reg64(read_regs[0].idx));
                        } else if (op.def_rc == jit_kernel_ir::RegisterClass::Mask) {
                            // Predicate copy: kmovq (kmovw would drop lanes
                            // above 16 for 8-bit element masks).
                            kmovq(Xbyak::Opmask(def_reg->idx),
                                  Xbyak::Opmask(read_regs[0].idx));
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
