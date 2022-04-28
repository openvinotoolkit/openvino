// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roll_kernel_selector.h"

#include "roll_kernel_ref.h"

namespace kernel_selector {

roll_kernel_selector::roll_kernel_selector() {
    Attach<RollKernelRef>();
}

KernelsData roll_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::ROLL);
}

}  // namespace kernel_selector
