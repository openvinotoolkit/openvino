// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc_kernel_selector.hpp"

#include "histc_kernel_ref.hpp"

namespace kernel_selector {

histc_kernel_selector::histc_kernel_selector() {
    Attach<HistcKernelRef>();
}

KernelsData histc_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::HISTC);
}

histc_kernel_selector& histc_kernel_selector::Instance() {
    static histc_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
