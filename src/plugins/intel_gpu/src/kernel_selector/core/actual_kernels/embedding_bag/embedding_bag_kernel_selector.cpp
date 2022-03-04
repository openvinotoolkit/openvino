// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_kernel_selector.h"
#include "embedding_bag_kernel_ref.h"

namespace kernel_selector {

embedding_bag_kernel_selector::embedding_bag_kernel_selector() {
    Attach<EmbeddingBagKernelRef>();
}

KernelsData embedding_bag_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::EMBEDDING_BAG);
}
}  // namespace kernel_selector
