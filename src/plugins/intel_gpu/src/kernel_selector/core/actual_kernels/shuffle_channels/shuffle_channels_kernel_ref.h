// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
