// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth_kernel_selector.h"
#include "space_to_depth_kernel_ref.h"

namespace kernel_selector {

    space_to_depth_kernel_selector::space_to_depth_kernel_selector() { Attach<SpaceToDepthKernelRef>(); }

    KernelsData space_to_depth_kernel_selector::GetBestKernels(const Params& params) const {
        return GetNaiveBestKernel(params, KernelType::SPACE_TO_DEPTH);
    }
}  // namespace kernel_selector
