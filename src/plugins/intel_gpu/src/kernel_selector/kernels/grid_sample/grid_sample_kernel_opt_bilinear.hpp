// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "grid_sample_kernel_base.hpp"

namespace kernel_selector {

class GridSampleKernelOptBilinear : public GridSampleKernelBase {
public:
    GridSampleKernelOptBilinear() : GridSampleKernelBase("grid_sample_opt_bilinear") {}

protected:
    CommonDispatchData CalcDispatch(const grid_sample_params& kernel_params) const override;
    KernelsPriority GetKernelsPriority(const Params& /*params*/) const override;
};

}  // namespace kernel_selector
