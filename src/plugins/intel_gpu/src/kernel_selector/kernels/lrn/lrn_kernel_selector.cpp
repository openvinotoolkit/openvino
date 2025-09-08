// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_selector.h"
#include "lrn_kernel_ref.h"
#include "lrn_kernel_within_channel_ref.h"
#include "lrn_kernel_within_channel_ref_opt.h"
#include "lrn_kernel_across_channel_ref.h"
#include "lrn_kernel_across_channel_opt_b8.h"
#include "lrn_kernel_across_channel_multiple_features.h"
#include "lrn_kernel_across_channel_multiple_features_fsv16.h"
#include "lrn_kernel_within_channel_byxf_opt.h"

namespace kernel_selector {
lrn_kernel_selector::lrn_kernel_selector() {
    Attach<LRNKernelRef>();
    Attach<LRNKernelWithinChannel>();
    Attach<LRNKernelWithinChannelOpt>();
    Attach<LRNKernelAcrossChannelRef>();
    Attach<LRNKernelAcrossChannel_b8>();
    Attach<LRNKernelWithinChannelByxfOpt>();
    Attach<LRNKernelAcrossChannelMultipleFeatures>();
    Attach<LRNKernelAcrossChannelMultipleFeaturesFSV16>();
}

KernelsData lrn_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::LRN);
}
}  // namespace kernel_selector
