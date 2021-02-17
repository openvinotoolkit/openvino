// Copyright (c) 2020 Intel Corporation
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


#pragma once

#include "lrn_kernel_across_channel_multiple_features.h"

namespace kernel_selector {
class LRNKernelAcrossChannelMultipleFeaturesFSV16 : public LRNKernelAcrossChannelMultipleFeatures {
public:
    using Parent = LRNKernelAcrossChannelMultipleFeatures;
    LRNKernelAcrossChannelMultipleFeaturesFSV16() : Parent("lrn_gpu_across_channel_multiple_features_fsv16") {}

    ParamsKey GetSupportedKey() const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

private:
    DispatchData SetDefault(const lrn_params& params) const override;
    JitConstants GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
