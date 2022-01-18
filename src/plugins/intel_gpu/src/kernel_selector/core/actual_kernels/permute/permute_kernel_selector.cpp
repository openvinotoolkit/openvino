// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_selector.h"
#include "permute_kernel_ref.h"
#include "permute_kernel_tile_8x8_4x4.h"
#include "permute_kernel_tile_8x8_4x4_fsv.h"

namespace kernel_selector {

permute_kernel_selector::permute_kernel_selector() {
    Attach<PermuteKernelRef>();
    Attach<PermuteKernel_tile_8x8_4x4>();
    Attach<PermuteKernel_tile_8x8_4x4_fsv>();
}

KernelsData permute_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::PERMUTE);
}
}  // namespace kernel_selector
