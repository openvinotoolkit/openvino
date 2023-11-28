// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// concatenation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct concatenation_params : public base_params {
    concatenation_params() : base_params(KernelType::CONCATENATION) {}

    ConcatAxis axis = ConcatAxis::FEATURE;
    bool isAligned = true;
    size_t misalignment = 0;

    size_t kernel_split_id = 0;
    MultiDataTensor original_input_layouts;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnableConcatAxis(axis);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// concatenation_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct concatenation_optional_params : optional_params {
    concatenation_optional_params() : optional_params(KernelType::CONCATENATION) {}
    bool kernelPerInput = true;

    ParamsKey GetSupportedKey() const override {
        ParamsKey k = optional_params::GetSupportedKey();

        if (kernelPerInput) {
            k.EnableConcatKernelPerInput();
        } else {
            k.EnableConcatOneKernel();
        }

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ConcatenationKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ConcatenationKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ConcatenationKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const concatenation_params& params) const;
    virtual DispatchData SetDefault(const concatenation_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    int32_t GetConcatChannelIndex(const concatenation_params& params) const;
    Tensor::DataChannelName GetConcatChannel(const concatenation_params& params) const;
    virtual size_t GetAlignment(const concatenation_params& /*params*/) const {
        return 1;
    }
    void SetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
