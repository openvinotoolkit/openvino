/*
// Copyright (c) 2019 Intel Corporation
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
*/

#pragma once

#include "common_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shuffle_channels_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct shuffle_channels_params : public base_params {
    shuffle_channels_params() : base_params(KernelType::SHUFFLE_CHANNELS) {}

    int32_t group;
    int32_t axis;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shuffle_channels_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct shuffle_channels_optional_params : optional_params {
    shuffle_channels_optional_params() : optional_params(KernelType::SHUFFLE_CHANNELS) {}
};

class ShuffleChannelsKernelRef : public common_kernel_base {
public:
    ShuffleChannelsKernelRef() : common_kernel_base("shuffle_channels_ref") {}
    virtual ~ShuffleChannelsKernelRef() {}
    virtual JitConstants GetJitConstants(const shuffle_channels_params& params) const;
    virtual CommonDispatchData SetDefault(const shuffle_channels_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
