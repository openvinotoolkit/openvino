// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Target capability queries for the jit_kernel DSL — the equivalent of
// LLVM's TargetTransformInfo. Everything the *front end* has to know about
// the ISA before it records anything lives here: whether memory access can
// be predicated, whether interleaved access can be predicated, and how a
// loop should handle a trip count that is not a multiple of the vector
// width.
//
// Instruction emission does NOT live here — that belongs to the arch's
// jit_kernel (see jit_kernel.md, "Phase 3 — Arch generalization"). The
// split is deliberate: queries are answered before recording, emission
// happens after allocation.
//
// The interface is arch-neutral by construction; each of the ISAs we care
// about has a distinct answer:
//
//   AVX2/SSE : no usable predication for our type set -> epilogue
//   AVX-512  : k-register predication                 -> mask
//   NEON     : no predication                         -> epilogue
//   SVE      : governing predicates everywhere        -> mask
//   RVV      : vl set per iteration by vsetvli        -> length
//
// The `length` style is what RVV needs and is not expressible as a mask:
// `vl` is machine state, so a loop sets it once per iteration in the loop
// header and the memory operations encode nothing. LLVM models the same
// thing with an explicit vector length operand in IR (`llvm.vp.*`), and at
// MIR level with implicit VL/VTYPE register operands plus the
// RISCVInsertVSETVLI dataflow pass.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ov::intel_cpu {

struct vector_target {
    // How to handle a trip count that is not a multiple of the vector
    // width. Mirrors LLVM's TailFoldingStyle.
    enum class tail_folding : std::uint8_t {
        epilogue,  // main loop at full width + a separate tail
        mask,      // one loop, per-iteration predicate (AVX-512, SVE)
        length,    // one loop, per-iteration vector length (RVV vsetvli)
    };

    vector_target() = default;
    vector_target(const vector_target&) = delete;
    vector_target& operator=(const vector_target&) = delete;
    virtual ~vector_target() = default;

    // Can a load/store of `elem_bytes`-wide elements be predicated?
    // LLVM: TargetTransformInfo::isLegalMaskedLoad/isLegalMaskedStore.
    [[nodiscard]] virtual bool supports_masked_access(std::size_t elem_bytes) const = 0;

    // Can an interleaved (de)interleaving access be predicated? True on
    // SVE (ST3 with a governing predicate) and RVV (segment stores honour
    // vl); false on x86, which has to build the interleave by hand.
    // LLVM: TargetTransformInfo::enableMaskedInterleavedAccessVectorization.
    [[nodiscard]] virtual bool supports_masked_interleaved_access() const = 0;

    // LLVM: TargetTransformInfo::getPreferredTailFoldingStyle.
    [[nodiscard]] virtual tail_folding preferred_tail_folding() const = 0;

    // Can a load/store of `elem_bytes`-wide elements carry this constant
    // displacement in its addressing mode, or does the pointer have to be
    // incremented instead? Asked once per peeled iteration, so that
    // straight-line code addresses off one base instead of bumping four
    // pointers per body.
    //
    // LLVM: TargetLowering::isLegalAddressingMode, restricted to the
    // base+displacement form. The displacement arrives twice because the
    // targets disagree about what it even is:
    //
    //   x86   : `bytes`, signed 32-bit, free in the encoding
    //           (X86AddressMode::Disp)
    //   NEON  : `bytes`, 9-bit signed or size-scaled 12-bit unsigned
    //   SVE   : `vectors` only — [x, #imm, MUL VL], imm in -8..7. A byte
    //           offset is illegal for a scalable type, which is why this
    //           is not a byte-only query (AArch64ISelLowering, the
    //           ScalableOffset path)
    //   RVV   : neither. "RVV instructions only support register
    //           addressing" — a vector access takes a base register and
    //           nothing else, so the increment must stay (RISCVISelLowering)
    [[nodiscard]] virtual bool is_legal_access_offset(std::size_t elem_bytes,
                                                      std::size_t vectors,
                                                      std::size_t bytes) const = 0;

    // Preferred code alignment for a loop header, in bytes; 0 to leave
    // loops unaligned. LLVM: TargetLowering::getPrefLoopAlignment, which
    // X86 sets to 16 (X86ISelLowering.cpp: setPrefLoopAlignment(Align(16))).
    // oneDNN's BRGEMM kernels align their loops to 64, so the right value
    // is a question for measurement rather than for doctrine — hence the
    // OV_JIT_IR_LOOP_ALIGN override.
    [[nodiscard]] virtual std::size_t preferred_loop_alignment() const = 0;

    // Allocation order for the predicate register file: the physical
    // registers the allocator may use for Mask values, in preference
    // order. Empty when the ISA has no predicates (SSE, AVX2, NEON).
    //
    // This is where per-target predicate constraints live, the way LLVM
    // puts them in register classes: x86 omits k0 (it cannot encode a
    // write-mask — VK*WM), SVE would list p0-p7 for governing predicates
    // (PPR_3b), and RVV would list only v0 (VMV0).
    [[nodiscard]] virtual const std::vector<std::uint32_t>& predicate_pool() const = 0;
};

// `OV_JIT_TAIL_FOLDING=epilogue|mask|length` overrides the preferred tail
// folding style, mirroring LLVM's -prefer-predicate-over-epilogue. Meant
// for A/B measurement and for exercising the strategy the host would not
// otherwise pick.

// The target describing the host this process is generating code for.
const vector_target& host_vector_target();

}  // namespace ov::intel_cpu
