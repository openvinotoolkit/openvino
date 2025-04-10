// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalize_kernel_base.h"

namespace kernel_selector {
class NormalizeKernelAcrossSpatialRef : public NormalizeKernelBase {
public:
    NormalizeKernelAcrossSpatialRef() : NormalizeKernelBase("normalize_gpu_across_spatial_ref") {}
    virtual ~NormalizeKernelAcrossSpatialRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
