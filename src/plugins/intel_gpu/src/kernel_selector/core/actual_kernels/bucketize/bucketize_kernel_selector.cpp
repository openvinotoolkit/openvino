// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize_kernel_selector.h"
#include "bucketize_kernel_ref.h"

namespace kernel_selector {

bucketize_kernel_selector::bucketize_kernel_selector() {
    Attach<BucketizeKernelRef>();
}

KernelsData bucketize_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::BUCKETIZE);
}
}  // namespace kernel_selector
