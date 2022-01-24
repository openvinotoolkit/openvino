// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_selector.h"
#include "arg_max_min_kernel_gpu_ref.h"
#include "arg_max_min_kernel_opt.h"
#include "arg_max_min_kernel_axis.h"

namespace kernel_selector {

arg_max_min_kernel_selector::arg_max_min_kernel_selector() {
    Attach<ArgMaxMinKernelGPURef>();
    // Attach<ArgMaxMinKernelOpt>(); not yet implemented
    Attach<ArgMaxMinKernelAxis>();
}

KernelsData arg_max_min_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ARG_MAX_MIN);
}
}  // namespace kernel_selector