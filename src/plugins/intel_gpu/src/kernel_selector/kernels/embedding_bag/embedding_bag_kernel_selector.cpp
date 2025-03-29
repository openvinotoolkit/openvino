// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_kernel_selector.h"
#include "embedding_bag_kernel_ref.h"

namespace kernel_selector {

embedding_bag_kernel_selector::embedding_bag_kernel_selector() {
    Attach<EmbeddingBagKernelRef>();
}

KernelsData embedding_bag_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::EMBEDDING_BAG);
}
}  // namespace kernel_selector
