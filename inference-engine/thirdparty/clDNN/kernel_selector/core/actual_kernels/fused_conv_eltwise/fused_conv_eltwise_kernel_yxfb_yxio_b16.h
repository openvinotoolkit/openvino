// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fused_conv_eltwise_kernel_base.h"
#include <string>
#include <vector>

namespace kernel_selector {

class fused_conv_eltwise_kernel_yxfb_yxio_b16 : public fused_conv_eltwise_kernel_base {
public:
    using Parent = fused_conv_eltwise_kernel_base;
    fused_conv_eltwise_kernel_yxfb_yxio_b16()
        : fused_conv_eltwise_kernel_base("fused_conv_eltwise_gpu_yxfb_yxio_b16") {}
    virtual ~fused_conv_eltwise_kernel_yxfb_yxio_b16() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferreddWeightsLayout(const fused_conv_eltwise_params &) const override {
        return WeightsLayout::yxio;
    }
    std::string GetKernelName(const fused_conv_eltwise_params&) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const fused_conv_eltwise_params& params, const DispatchData& dispatchData) const override;
    DispatchData SetDefault(const fused_conv_eltwise_params& arg, int autoTuneIndex = -1) const override;
};
}  // namespace kernel_selector
