// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generate_proposals_kernel_selector.h"
#include "generate_proposals_kernel_ref.h"

namespace kernel_selector {
generate_proposals_kernel_selector::generate_proposals_kernel_selector() {
    Attach<GenerateProposalsRef>();
}

generate_proposals_kernel_selector& generate_proposals_kernel_selector::Instance() {
    static generate_proposals_kernel_selector instance_;
    return instance_;
}

KernelsData generate_proposals_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::GENERATE_PROPOSALS);
}
}  // namespace kernel_selector
