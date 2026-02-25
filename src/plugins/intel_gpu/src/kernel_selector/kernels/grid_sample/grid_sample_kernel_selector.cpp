// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel_selector.hpp"

#include "grid_sample_kernel_opt_bilinear_zeros.hpp"
#include "grid_sample_kernel_ref.hpp"

namespace kernel_selector {

grid_sample_kernel_selector::grid_sample_kernel_selector() {
    Attach<GridSampleKernelOpt_BilinearZeros>();
    Attach<GridSampleKernelRef>();
}

KernelsData grid_sample_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GRID_SAMPLE);
}

grid_sample_kernel_selector& grid_sample_kernel_selector::Instance() {
    static grid_sample_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
