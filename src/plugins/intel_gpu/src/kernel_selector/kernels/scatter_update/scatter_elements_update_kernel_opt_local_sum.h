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
// SUM-reduction fast path. Accumulates into an int32 fixed-point buffer in three stages,
// with each workgroup staging its contributions in a local-memory window before flushing
// them. A destination outside the window takes the same global atomic `_ref` uses, so the
// window affects effectiveness only. Encoding and accumulator semantics are `_ref`'s.
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
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    static constexpr size_t kWindowSize = 4096;  // 16KB of local memory

    // Shape-agnostic builds carry both update variants and skip one per resolved shape.
    static constexpr size_t kShapeAgnosticKernelCount = 4;
    static constexpr size_t kStageUpdateAnchored = 1;
    static constexpr size_t kStageUpdateFromZero = 2;

    static size_t GetAccumulatorSize(const DataTensor& output);
    // Valid only if every destination index falls inside a window anchored at 0.
    static bool AnchorAtZero(const DataTensor& output) { return output.PhysicalSize() <= kWindowSize; }
};
}  // namespace kernel_selector
