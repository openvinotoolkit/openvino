// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "grid_sample_kernel_base.hpp"

namespace kernel_selector {

/**
 * Reference kernel for GridSample.
 */
class GridSampleKernelRef : public GridSampleKernelBase {
public:
    GridSampleKernelRef() : GridSampleKernelBase("grid_sample_ref") {}

protected:
    ParamsKey GetSupportedKey() const override;
    CommonDispatchData CalcDispatch(const grid_sample_params& kernel_params) const override;
};

}  // namespace kernel_selector
