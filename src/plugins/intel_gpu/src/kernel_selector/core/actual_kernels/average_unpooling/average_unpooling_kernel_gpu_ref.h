// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "average_unpooling_kernel_base.h"

namespace kernel_selector {
class AverageUnpoolingKernelGPURef : public AverageUnpoolingKernelBase {
public:
    AverageUnpoolingKernelGPURef() : AverageUnpoolingKernelBase("average_unpooling_gpu_ref") {}
    virtual ~AverageUnpoolingKernelGPURef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
