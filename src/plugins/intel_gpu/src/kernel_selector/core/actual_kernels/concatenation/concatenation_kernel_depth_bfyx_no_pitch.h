// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "concatenation_kernel_base.h"

namespace kernel_selector {

class ConcatenationKernel_depth_bfyx_no_pitch : public ConcatenationKernelBase {
public:
    ConcatenationKernel_depth_bfyx_no_pitch() : ConcatenationKernelBase("concatenation_gpu_depth_bfyx_no_pitch") {}
    virtual ~ConcatenationKernel_depth_bfyx_no_pitch() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DispatchData SetDefault(const concatenation_params& params) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
