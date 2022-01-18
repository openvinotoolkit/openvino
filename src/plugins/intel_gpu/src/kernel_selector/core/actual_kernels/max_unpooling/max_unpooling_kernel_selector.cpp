// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_unpooling_kernel_selector.h"
#include "max_unpooling_kernel_gpu_ref.h"

namespace kernel_selector {

max_unpooling_kernel_selector::max_unpooling_kernel_selector() { Attach<MaxUnpoolingKernelGPURef>(); }

KernelsData max_unpooling_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::MAX_UNPOOLING);
}
}  // namespace kernel_selector