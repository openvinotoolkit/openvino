// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <kernel_base_opencl.h>

namespace kernel_selector {

enum class roi_aligned_mode : uint32_t { ASYMMETRIC, HALF_PIXEL_FOR_NN, HALF_PIXEL };

struct roi_align_params : public base_params {
    roi_align_params() : base_params{KernelType::ROI_ALIGN} {}

    int sampling_ratio = 0;
    float spatial_scale = 1.f;
    PoolType pooling_mode = PoolType::MAX;
    roi_aligned_mode aligned_mode = roi_aligned_mode::ASYMMETRIC;
    bool rotated_mode = false;
    bool clockwise = false;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnablePoolType(pooling_mode);
        return k;
    }
};

class ROIAlignKernelRef : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

    ROIAlignKernelRef() : KernelBaseOpenCL{"roi_align_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params&) const override;

protected:
    JitConstants GetJitConstants(const roi_align_params& params) const;
};

} // namespace kernel_selector
