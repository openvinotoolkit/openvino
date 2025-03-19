// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalize_kernel_base.h"

namespace kernel_selector {
class NormalizeKernelWithinSpatialRef : public NormalizeKernelBase {
public:
    NormalizeKernelWithinSpatialRef() : NormalizeKernelBase("normalize_gpu_within_spatial_ref") {}
    virtual ~NormalizeKernelWithinSpatialRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
