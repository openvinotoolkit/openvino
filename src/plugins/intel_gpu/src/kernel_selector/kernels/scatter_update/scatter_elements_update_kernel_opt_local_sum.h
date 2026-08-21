// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "scatter_elements_update_kernel_ref.h"  // reuse scatter_elements_update_params only

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ScatterElementsUpdateKernelOptLocalSum
//
// A narrowly-eligible fast path for the SUM-reduction, dense-scatter case (updates
// tensor the same total size as the output -- e.g. a bilinear-splat forward warp),
// where the existing `_ref` kernel's global-memory atomic-accumulate stage dominates
// total cost. Real hardware measurement (Arc B70, 1920x1088xf16, DRBA softsplat)
// isolated the per-stage cost: init ~40-60us, atomic-accumulate ~5.6-9.3ms, finalize
// ~45-85us -- the accumulate stage alone is ~99% of the primitive's total time.
// A synthetic index-pattern precheck (identity/heavy-collision/random/extreme-collision,
// same element count) showed raw collision *count* barely matters (heavy-collision ~=
// identity, within noise) but *memory access locality* does (random ~4.6x slower than
// identity, despite similar average collision rate) -- the real cost driver is DRAM
// transaction coalescing on scattered global-memory atomics, not contention per se.
//
// This kernel stages each workgroup's contributions in a small, fixed-size local
// (on-chip) memory window before flushing to global memory, converting scattered
// global atomics into local atomics (architecturally cheap and coalescing-independent)
// followed by one clustered, mostly-nonzero-only global write-back per workgroup. Any
// destination index that falls outside a workgroup's window (large/divergent motion,
// no assumption of bounded flow is made or required) falls back to the *exact same*
// direct global atomic_fetch_add the `_ref` kernel already does for every element --
// so this can never be less correct than `_ref`, only potentially less effective.
//
// Correctness rests on one exact (not approximate) invariant: the SUM-mode accumulator
// is fixed-point int32 (see this kernel's own to_int/from_int, copied unchanged from
// `_ref`'s), so integer addition is truly associative/commutative -- staging some
// contributions locally before adding them to global memory changes only the order of
// additions, never the result (barring the same int32 overflow risk the reference
// kernel already has). Validated bit-identical against `_ref` on the existing GPU unit
// test suite plus new coverage for the local/global-fallback boundary (see the PR test
// additions).
//
// Deliberately independent of ScatterElementsUpdateKernelRef (no inheritance) -- purely
// additive, touches nothing in the existing, already-validated kernel. Only attached
// ahead of `_ref` in the selector; Validate() returning false for anything outside this
// kernel's narrow scope falls straight through to `_ref` unchanged.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ScatterElementsUpdateKernelOptLocalSum : public KernelBaseOpenCL {
public:
    ScatterElementsUpdateKernelOptLocalSum() : KernelBaseOpenCL("scatter_elements_update_opt_local_sum") {}
    ~ScatterElementsUpdateKernelOptLocalSum() override = default;

    JitConstants GetJitConstants(const scatter_elements_update_params& params) const;
    CommonDispatchData SetDefault(const scatter_elements_update_params& params, bool is_second) const;
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;

protected:
    bool Validate(const Params& p) const override;
    bool SkipKernelExecution(const scatter_elements_update_params& params, size_t kernel_id) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    // 16KB local memory, comfortably under the ~64KB typical per-workgroup budget.
    static constexpr size_t kWindowSize = 4096;
};
}  // namespace kernel_selector
