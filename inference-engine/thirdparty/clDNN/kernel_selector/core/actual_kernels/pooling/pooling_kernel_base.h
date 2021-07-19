// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// pooling_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct pooling_params : public base_params {
    pooling_params() : base_params(KernelType::POOLING) {}

    PoolType poolType = PoolType::MAX;
    PoolRemainder remainderAction = PoolRemainder::FLOOR;
    KernelDividerMode divMode = KernelDividerMode::DONT_CARE;
    QuantizationType quantization = QuantizationType::SYMMETRIC;
    uSize poolSize;
    uSize poolStride;
    uSize poolPad;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        k.EnablePoolType(poolType);
        k.EnablePoolRemainder(remainderAction);
        k.EnablePoolKernelDividerMode(divMode);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// pooling_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct pooling_optional_params : optional_params {
    pooling_optional_params() : optional_params(KernelType::POOLING) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PoolingKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PoolingKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~PoolingKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        bool needsBoundary;
    };

protected:
    bool Validate(const Params&, const optional_params&) const override;
    virtual JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const pooling_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
    Datatype GetAccumulatorType(const pooling_params& p) const;
    Datatype GetActivationType(const pooling_params& params) const;
    bool NeedsBoundaryCheck(const pooling_params& params) const;
    bool EnableRound(const pooling_params& params) const;
};
}  // namespace kernel_selector
