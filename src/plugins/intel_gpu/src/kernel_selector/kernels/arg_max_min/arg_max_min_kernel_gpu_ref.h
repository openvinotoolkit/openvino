// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arg_max_min_kernel_base.h"

namespace kernel_selector {
class ArgMaxMinKernelGPURef : public ArgMaxMinKernelBase {
public:
    ArgMaxMinKernelGPURef() : ArgMaxMinKernelBase("arg_max_min_gpu_ref") {}
    virtual ~ArgMaxMinKernelGPURef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
