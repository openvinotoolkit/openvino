// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "resample_kernel_base.h"
#include <string>

namespace kernel_selector {
class ResampleKernelPilRef : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    enum KernelId {
        eCalcHorizontalCoefficients,
        eResampleHorizontal,
        eCalcVerticalCoefficients,
        eResampleVertical,
        eEnd
    };

    ResampleKernelPilRef() : ResampleKernelBase(std::string{"resample_pil_ref"}) {}

    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstantsForKernel(KernelId id, const resample_params& params) const;
    DispatchData SetDefaultForKernel(KernelId id, const resample_params &arg) const;
};
}  // namespace kernel_selector
