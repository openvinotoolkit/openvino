// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_tree_kernel_selector.h"
#include "gather_tree_kernel_ref.h"

namespace kernel_selector {
    gather_tree_kernel_selector::gather_tree_kernel_selector() { Attach<GatherTreeKernelRef>(); }

    KernelsData gather_tree_kernel_selector::GetBestKernels(const Params& params,
                                                            const optional_params& options) const {
        return GetNaiveBestKernel(params, options, KernelType::GATHER_TREE);
    }
}  // namespace kernel_selector
