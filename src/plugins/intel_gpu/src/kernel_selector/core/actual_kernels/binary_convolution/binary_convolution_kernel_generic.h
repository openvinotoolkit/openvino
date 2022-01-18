// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "binary_convolution_kernel_base.h"
#include <vector>

namespace kernel_selector {

class BinaryConvolutionKernelGeneric : public BinaryConvolutionKernelBase {
public:
    using Parent = BinaryConvolutionKernelBase;

    BinaryConvolutionKernelGeneric() : BinaryConvolutionKernelBase("binary_convolution_gpu_generic") {}
    virtual ~BinaryConvolutionKernelGeneric() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    WeightsLayout GetPreferredWeightLayout(const binary_convolution_params &) const override {
        return WeightsLayout::os_is_yx_osv32_isv32p;
    }
    JitConstants GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                const DispatchData& dispatchData) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    DispatchData SetDefault(const binary_convolution_params& arg, int autoTuneIndex = -1) const override;
    JitConstants GetJitConstants(const binary_convolution_params& params, const DispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
