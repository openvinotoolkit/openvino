// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_selector.h"
#include "quantize_kernel_ref.h"
#include "quantize_kernel_scale_shift_opt.h"

namespace kernel_selector {

quantize_kernel_selector::quantize_kernel_selector() {
    Attach<QuantizeKernelRef>();
    Attach<QuantizeKernelScaleShift>();
}

KernelsData quantize_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::QUANTIZE);
}
}  // namespace kernel_selector
