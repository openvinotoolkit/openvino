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

    // Can a vector instruction take one `elem_bytes` element from memory
    // and splat it across the vector, in place of a full-width source
    // operand? AVX-512's embedded broadcast, `{1to16}` in Intel syntax.
    //
    // Asked before a broadcast is folded into its consumer. LLVM does not
    // need a query here because it does not need a decision: the
    // broadcast fold tables name EVEX opcodes, and a subtarget without
    // AVX-512 never has an instruction those entries apply to. This IR's
    // fold closures are written once for all x86, so the availability has
    // to be asked rather than fall out of instruction selection.
    //
    // Not one query per operand shape: what varies between targets is
    // whether the form exists at all. SVE has no equivalent — a splatted
    // operand is a separate DUP — and NEON's by-element multiply is a
    // lane index rather than a memory operand.
    [[nodiscard]] virtual bool supports_broadcast_memory_operand(std::size_t elem_bytes) const = 0;

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

    // How many padding bytes an alignment may spend; 0 means unlimited.
    // LLVM: TargetLowering::getMaxPermittedBytesForAlignment, read by
    // MachineBlockPlacement::alignBlocks and passed down to the
    // AsmPrinter as the second operand of .p2align.
    //
    // TargetLoweringBase initializes it to 0 and only AArch64 and
    // LoongArch override it, so X86 aligns loops to 16 with no cap. A cap
    // is not a free safety margin: it silently drops the alignment on
    // exactly those loops that need the most padding, which is
    // uncorrelated with how hot they are. Capping at half the alignment
    // left two of the three BRGEMM loops unaligned, including both inner
    // ones.
    [[nodiscard]] virtual std::size_t max_bytes_for_alignment() const = 0;

    // Cache line size in bytes, and how far ahead a streaming read
    // should be prefetched, also in bytes; 0 for either means do not
    // prefetch. LLVM: TargetTransformInfo::getCacheLineSize and
    // getPrefetchDistance, which together gate its LoopDataPrefetch pass
    // (LoopDataPrefetch.cpp: "If PrefetchDistance is not set, don't run
    // the pass").
    //
    // X86TargetTransformInfo answers neither, so LLVM never
    // software-prefetches on x86, trusting the hardware prefetcher for
    // strided access. This target deliberately disagrees, and the
    // divergence is the one place in this file where measurement beat
    // doctrine: on the MatMul BRGEMM benchmark, prefetching B one
    // reduction block ahead is worth 5.5% of the kernel's cycles per FMA
    // (median 0.2960 against 0.3133 without, oneDNN 0.2999), which is
    // the entire gap against oneDNN's hand-written kernel.
    //
    // The first attempt at this question concluded the opposite, from
    // wall-clock A/B on a whole inference: 84/87/85 us against 85/85/88.
    // That instrument cannot see a 3% difference in a kernel that is
    // part of an 80 us number with 3 us of run-to-run drift. The lesson
    // is about the metric, not about prefetching — see the journal.
    //
    // LLVM's units here are instructions, not bytes: LoopDataPrefetch
    // derives the address as ItersAhead * stride with ItersAhead =
    // PrefetchDistance / LoopSize. With no such pass, a recording site
    // knows its own stride, so the target answers in bytes and the site
    // rounds up to a whole number of loop strides.
    //
    // OV_JIT_IR_PREFETCH overrides the distance, 0 to turn it off.
    [[nodiscard]] virtual std::size_t cache_line_size() const = 0;
    [[nodiscard]] virtual std::size_t prefetch_distance() const = 0;

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
