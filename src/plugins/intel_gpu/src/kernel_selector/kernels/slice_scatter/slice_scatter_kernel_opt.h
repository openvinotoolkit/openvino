// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "slice_scatter_kernel_ref.h"

namespace kernel_selector {

class SliceScatterKernelOpt : public KernelBaseOpenCL {
public:
    SliceScatterKernelOpt() : KernelBaseOpenCL{"slice_scatter_opt"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

private:
    JitConstants GetJitConstants(const slice_scatter_params& params) const;
    CommonDispatchData SetDefault(const slice_scatter_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    static constexpr size_t VEC_SIZE = 8;
    static constexpr size_t SIMD_SIZE = 16;
};

}  // namespace kernel_selector
