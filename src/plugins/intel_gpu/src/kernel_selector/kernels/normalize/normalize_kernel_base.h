// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// normalize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct normalize_params : public base_params {
    normalize_params() : base_params(KernelType::NORMALIZE) {}

    NormalizeMode normMode = NormalizeMode::ACROSS_SPATIAL;
    float epsilon = 1e-10f;
    DataTensor scaleTable;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        k.EnableNormalizeMode(normMode);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NormalizeKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NormalizeKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~NormalizeKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const normalize_params& params) const;
    DispatchData SetDefault(const normalize_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }
    bool Validate(const Params& params) const override;
    Datatype GetActivationType(const normalize_params& params) const;
};
}  // namespace kernel_selector
