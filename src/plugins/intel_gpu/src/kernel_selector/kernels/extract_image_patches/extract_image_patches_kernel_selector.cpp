// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_kernel_selector.h"
#include "extract_image_patches_kernel_ref.h"

namespace kernel_selector {
extract_image_patches_kernel_selector::extract_image_patches_kernel_selector() {
    Attach<ExtractImagePatchesKernelRef>();
}

KernelsData extract_image_patches_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::EXTRACT_IMAGE_PATCHES);
}
}  // namespace kernel_selector
