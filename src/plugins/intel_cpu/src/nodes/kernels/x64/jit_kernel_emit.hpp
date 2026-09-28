// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// The instruction-emission interface the portable DSL constructs are
// written against — the equivalent of LLVM's TargetInstrInfo, and the
// counterpart of jit_kernel_target.hpp: that header answers questions
// before anything is recorded, this one emits.
//
// Every primitive is a *factory* for an emit closure, not an emitter. The
// portable layer keeps all the IR modelling — which operands are read,
// which is tied, whether the def is early-clobber, which register class it
// belongs to — because that is where correctness lives and none of it is
// arch-specific. The arch supplies only which instruction realizes the
// operation, and receives its operands as already-allocated registers
// through EmitContext.
//
// The set is deliberately small: it is exactly what the portable
// constructs (`foreach_vec`, the active-length realization, `ir_ptr`
// displacement arithmetic) call, derived by reading their bodies rather
// than by guessing what an architecture might want. Vector operations
// that only kernels call — permutes, shuffles, conversions, interleaves —
// stay on the arch's generator until a second target needs them.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "jit_kernel_ir.hpp"

namespace ov::intel_cpu {

// Portable condition codes. Only the ones the DSL actually branches on.
// LLVM keeps the same idea arch-neutral at IR level (ICmp predicates) and
// lowers it per target.
enum class cond : std::uint8_t {
    equal,
    not_equal,
    greater_equal,  // signed
};

// An opaque branch target. Owned by the arch (Xbyak::Label on x86, a
// different type elsewhere), so the portable layer only ever holds and
// passes the handle.
struct arch_label {
    arch_label() = default;
    arch_label(const arch_label&) = delete;
    arch_label& operator=(const arch_label&) = delete;
    virtual ~arch_label() = default;
};

using label_ref = std::shared_ptr<arch_label>;

struct arch_emitter {
    arch_emitter() = default;
    arch_emitter(const arch_emitter&) = delete;
    arch_emitter& operator=(const arch_emitter&) = delete;
    virtual ~arch_emitter() = default;

    // ── Scalar (GPR) arithmetic ───────────────────────────────────────
    // Operand naming follows the IR: `def` is the defined value, `reads`
    // are the read operands, in the order the recording site listed them.

    // def = reads[0]
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_copy() const = 0;

    // def = imm
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_set(std::uint64_t imm) const = 0;

    // def += imm, def -= imm, def >>= imm, def &= imm. Recorded with the
    // def tied to reads[0], so the portable layer — not the arch — is what
    // guarantees they share a register.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_add_imm(std::uint64_t imm) const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_shr_imm(unsigned shift) const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_and_imm(std::uint64_t imm) const = 0;

    // def = reads[0] * imm, non-destructive.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_imul_imm(std::uint64_t imm) const = 0;

    // reads[0] += delta, in place, with no def. The form a pointer advance
    // and a remaining-count decrement take: the value is mutated but the
    // IR declares only a read, which is why no pass may move a memory
    // access across one.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_bump(std::int64_t delta) const = 0;

    // def = reads[0] + imm, leaving reads[0] alone.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_offset(std::size_t imm) const = 0;

    // def = *(reads[0] + imm), a pointer-sized load. Recorded with
    // may_load so no pass moves it across a store.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_load(std::size_t imm) const = 0;

    // Set the condition state from reads[0] against an immediate or
    // against reads[1]. No def: the flags are not modelled as a value, so
    // the op is the side-effecting form and nothing may be scheduled
    // between it and the branch that consumes it.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_cmp_imm(std::uint64_t imm) const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn gpr_cmp_reg() const = 0;

    // ── Active length and predicates ──────────────────────────────────

    // def = min(reads[0], lanes). Recorded early-clobber: the expansion
    // writes its destination before consuming the count.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn clamped_len(std::size_t lanes) const = 0;

    // def = reads[0] >= lanes ? all-ones : (1 << reads[0]) - 1, in a GPR.
    // Also early-clobber. Split from the predicate itself because that is
    // how the hardware works: compute the bits, then move them into the
    // predicate file. On a target with a native active-lane-mask
    // instruction (SVE's whilelt) this returns a no-op closure and
    // `mask_from_bits` does the work.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn lane_mask_bits(std::size_t lanes) const = 0;

    // def (Mask class) = the low `lanes` bits of reads[0].
    [[nodiscard]] virtual jit_kernel_ir::EmitFn mask_from_bits(std::size_t lanes) const = 0;

    // ── Control flow ──────────────────────────────────────────────────

    // Bring *(reads[0] + imm) towards the core. `locality` is LLVM's
    // llvm.prefetch operand, not an x86 mnemonic: 0 means the data is
    // used once and should not displace anything (nta), 3 means keep it
    // in every level (t0).
    //
    // No def and no memory effect: a prefetch cannot fault and cannot be
    // observed, so it is kept only because def-less ops are never
    // eliminated.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn prefetch(std::size_t imm,
                                                          unsigned locality) const = 0;

    // Pad to a `bytes` boundary, spending at most `max_padding` bytes.
    // The counterpart of LLVM's AsmPrinter emitting .p2align for a block
    // whose MachineBasicBlock::Alignment is set: lowering reads the
    // property off the op and asks the arch to realize it.
    [[nodiscard]] virtual jit_kernel_ir::EmitFn align_to(std::size_t bytes,
                                                         std::size_t max_padding) const = 0;

    [[nodiscard]] virtual label_ref make_label() const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn place_label(const label_ref& at) const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn branch(cond on, const label_ref& to) const = 0;
    [[nodiscard]] virtual jit_kernel_ir::EmitFn branch_always(const label_ref& to) const = 0;
};

}  // namespace ov::intel_cpu
