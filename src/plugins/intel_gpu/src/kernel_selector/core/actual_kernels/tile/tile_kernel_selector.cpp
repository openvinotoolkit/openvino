// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_kernel_selector.h"
#include "tile_kernel_ref.h"

namespace kernel_selector {

tile_kernel_selector::tile_kernel_selector() { Attach<TileKernelRef>(); }

KernelsData tile_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::TILE);
}
}  // namespace kernel_selector
