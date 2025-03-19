// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_kernel_base.h"
#include "kernel_selector_params.h"
#include <string>
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// deconvolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct deconvolution_params : public weight_bias_params {
    deconvolution_params() : weight_bias_params(KernelType::DECONVOLUTION) {}

    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    uint32_t groups = 1;

    std::string to_string() const override;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        if (dilation.x != 1 || dilation.y != 1 || dilation.z != 1) {
            k.EnableDilation();
        }

        if (groups > 1) {
            k.EnableGroupedConvolution();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DeconvolutionKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DeconvolutionKernelBase : public WeightBiasKernelBase {
public:
    using WeightBiasKernelBase::WeightBiasKernelBase;
    virtual ~DeconvolutionKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    KernelsData GetKernelsData(const Params& params) const override;
    using WeightBiasKernelBase::GetJitConstants;
    virtual JitConstants GetJitConstants(const deconvolution_params& params) const;
    virtual DispatchData SetDefault(const deconvolution_params& params) const;
    virtual WeightsLayout GetPreferredWeightsLayout(const deconvolution_params &params) const {
        if (params.inputs[0].Dimentions() == 4)
            return (params.groups > 1) ? WeightsLayout::goiyx : WeightsLayout::oiyx;
        else
            return (params.groups > 1) ? WeightsLayout::goizyx : WeightsLayout::oizyx;
    }
    bool Validate(const Params& p) const override;

    virtual Datatype GetAccumulatorType(const deconvolution_params& params) const;
    virtual Datatype GetActivationType(const deconvolution_params& params) const;
};
}  // namespace kernel_selector
