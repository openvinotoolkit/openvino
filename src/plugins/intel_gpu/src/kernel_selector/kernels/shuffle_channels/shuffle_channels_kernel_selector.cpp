// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels_kernel_selector.h"
#include "shuffle_channels_kernel_ref.h"

namespace kernel_selector {

shuffle_channels_kernel_selector::shuffle_channels_kernel_selector() { Attach<ShuffleChannelsKernelRef>(); }

KernelsData shuffle_channels_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SHUFFLE_CHANNELS);
}
}  // namespace kernel_selector
