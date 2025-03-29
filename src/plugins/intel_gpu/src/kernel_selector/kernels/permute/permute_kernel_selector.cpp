// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_selector.h"
#include "permute_kernel_ref.h"
#include "permute_kernel_tile_8x8_4x4.h"
#include "permute_kernel_tile_8x8_4x4_fsv.h"
#include "permute_kernel_bfzyx_to_bfyxz.h"
#include "permute_kernel_f_y_axes.h"

namespace kernel_selector {

permute_kernel_selector::permute_kernel_selector() {
    Attach<PermuteKernelRef>();
    Attach<PermuteKernel_tile_8x8_4x4>();
    Attach<PermuteKernel_tile_8x8_4x4_fsv>();
    Attach<PermuteKernel_bfzyx_to_bfyxz>();
    Attach<PermuteKernel_f_y_axes>();
}

KernelsData permute_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PERMUTE);
}
}  // namespace kernel_selector
