// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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

KernelsData lrn_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::LRN);
}
}  // namespace kernel_selector
