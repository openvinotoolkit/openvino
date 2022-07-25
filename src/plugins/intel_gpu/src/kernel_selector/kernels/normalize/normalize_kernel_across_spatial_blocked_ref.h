// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "normalize_kernel_base.h"

namespace kernel_selector {
class NormalizeKernelAcrossSpatialBlockedRef : public NormalizeKernelBase {
public:
    NormalizeKernelAcrossSpatialBlockedRef() : NormalizeKernelBase("normalize_gpu_across_spatial_blocked_ref") {}
    virtual ~NormalizeKernelAcrossSpatialBlockedRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
