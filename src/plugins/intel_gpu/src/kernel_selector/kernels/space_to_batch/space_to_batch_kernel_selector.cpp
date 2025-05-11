// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_batch_kernel_selector.h"
#include "space_to_batch_kernel_ref.h"

namespace kernel_selector {

space_to_batch_kernel_selector::space_to_batch_kernel_selector() {
    Attach<SpaceToBatchKernelRef>();
}

KernelsData space_to_batch_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SPACE_TO_BATCH);
}
}  // namespace kernel_selector
