// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_kernel_selector.h"
#include "tile_kernel_ref.h"

namespace kernel_selector {

tile_kernel_selector::tile_kernel_selector() { Attach<TileKernelRef>(); }

KernelsData tile_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::TILE);
}
}  // namespace kernel_selector
