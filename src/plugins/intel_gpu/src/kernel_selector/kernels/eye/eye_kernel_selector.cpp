// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "eye_kernel_selector.h"

#include "eye_kernel_ref.h"

namespace kernel_selector {

eye_kernel_selector::eye_kernel_selector() {
    Attach<EyeKernelRef>();
}

KernelsData eye_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::EYE);
}

}  // namespace kernel_selector
