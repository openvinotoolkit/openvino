// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "edgpsi_kernel_selector.h"
#include "edgpsi_kernel_ref.h"

namespace kernel_selector {
edgpsi_kernel_selector::edgpsi_kernel_selector() {
    Attach<EDGPSIRef>();
}

edgpsi_kernel_selector& edgpsi_kernel_selector::Instance() {
    static edgpsi_kernel_selector instance_;
    return instance_;
}

KernelsData edgpsi_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE);
}
}  // namespace kernel_selector
