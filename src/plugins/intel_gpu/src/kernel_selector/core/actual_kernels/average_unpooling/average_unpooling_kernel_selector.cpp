// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "average_unpooling_kernel_selector.h"
#include "average_unpooling_kernel_gpu_ref.h"

namespace kernel_selector {

average_unpooling_kernel_selector::average_unpooling_kernel_selector() {
    Attach<AverageUnpoolingKernelGPURef>();
}

KernelsData average_unpooling_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::AVERAGE_UNPOOLING);
}
}  // namespace kernel_selector