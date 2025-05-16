// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_selector.h"
#include "lora_kernel_ref.h"
#include "lora_kernel_opt.h"

namespace kernel_selector {

lora_kernel_selector::lora_kernel_selector() {
    Attach<LoRAKernelRef>();
    Attach<LoRAKernelOpt>();
}

KernelsData lora_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LORA);
}
}  // namespace kernel_selector
