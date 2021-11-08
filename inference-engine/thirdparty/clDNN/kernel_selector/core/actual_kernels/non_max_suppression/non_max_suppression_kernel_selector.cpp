// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_max_suppression_kernel_selector.h"
#include "non_max_suppression_kernel_ref.h"

namespace kernel_selector {

non_max_suppression_kernel_selector::non_max_suppression_kernel_selector() { Attach<NonMaxSuppressionKernelRef>(); }

KernelsData non_max_suppression_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::NON_MAX_SUPPRESSION);
}
}  // namespace kernel_selector
