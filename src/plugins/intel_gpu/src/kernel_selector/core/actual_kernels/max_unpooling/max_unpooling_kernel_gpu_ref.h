// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "max_unpooling_kernel_base.h"

namespace kernel_selector {
class MaxUnpoolingKernelGPURef : public MaxUnpoolingKernelBase {
public:
    MaxUnpoolingKernelGPURef() : MaxUnpoolingKernelBase("max_unpooling_gpu_ref") {}
    virtual ~MaxUnpoolingKernelGPURef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
