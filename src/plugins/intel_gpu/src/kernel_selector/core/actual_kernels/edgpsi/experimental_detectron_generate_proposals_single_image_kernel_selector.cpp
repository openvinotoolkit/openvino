// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_generate_proposals_single_image_kernel_selector.h"
#include "experimental_detectron_generate_proposals_single_image_kernel_ref.h"

namespace kernel_selector {
experimental_detectron_generate_proposals_single_image_kernel_selector::experimental_detectron_generate_proposals_single_image_kernel_selector() {
    Attach<ExperimentalDetectronGenerateProposalsSingleImageRef>();
}

experimental_detectron_generate_proposals_single_image_kernel_selector& experimental_detectron_generate_proposals_single_image_kernel_selector::Instance() {
    static experimental_detectron_generate_proposals_single_image_kernel_selector instance_;
    return instance_;
}

KernelsData experimental_detectron_generate_proposals_single_image_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE);
}
}  // namespace kernel_selector
