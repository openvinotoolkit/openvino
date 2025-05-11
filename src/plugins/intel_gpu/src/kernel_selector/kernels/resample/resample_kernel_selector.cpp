// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_selector.h"
#include "resample_kernel_ref.h"
#include "resample_kernel_opt.h"
#include "resample_kernel_onnx.h"
#include "resample_kernel_pil_ref.h"

namespace kernel_selector {
resample_kernel_selector::resample_kernel_selector() {
    Attach<ResampleKernelRef>();
    Attach<ResampleKernelOpt>();
    Attach<ResampleKernelOnnx>();
    Attach<ResampleKernelPilRef>();
}

KernelsData resample_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::RESAMPLE);
}
}  // namespace kernel_selector
