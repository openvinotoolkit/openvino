/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// shuffle_channels_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct shuffle_channels_params : public base_params {
    shuffle_channels_params() : base_params(KernelType::SHUFFLE_CHANNELS), group(0), axis(0) {}

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

class ShuffleChannelsKernelRef : public KernelBaseOpenCL {
public:
    ShuffleChannelsKernelRef() : KernelBaseOpenCL("shuffle_channels_ref") {}
    virtual ~ShuffleChannelsKernelRef() {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual CommonDispatchData SetDefault(const shuffle_channels_params& params, const optional_params&) const;
    virtual JitConstants GetJitConstants(const shuffle_channels_params& params) const;
};
}  // namespace kernel_selector
