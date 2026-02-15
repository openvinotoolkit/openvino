// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "grid_sample_kernel_base.hpp"

namespace kernel_selector {

class GridSampleKernelOpt_BilinearZeros : public GridSampleKernelBase {
public:
    using TBase = GridSampleKernelBase;
    GridSampleKernelOpt_BilinearZeros() : GridSampleKernelBase("grid_sample_opt_bilinear_zeros") {}

protected:
    ParamsKey GetSupportedKey() const override;
    CommonDispatchData CalcDispatch(const grid_sample_params& kernel_params) const override;
    KernelsPriority GetKernelsPriority(const Params& /*params*/) const override;
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const grid_sample_params& kernel_params) const override;
};

}  // namespace kernel_selector
