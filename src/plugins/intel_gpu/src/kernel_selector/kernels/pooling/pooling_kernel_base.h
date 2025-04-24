// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_types.h"
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
    bool maxPoolOpset8Features = false;
    uSize poolDilation{1, 1, 1};
    Datatype poolIndexElementType = Datatype::INT64;
    int64_t poolAxis = 0;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();

        k.EnablePoolType(poolType);
        k.EnablePoolRemainder(remainderAction);
        k.EnablePoolKernelDividerMode(divMode);

        if (maxPoolOpset8Features) {
            k.EnablePoolDilation();
            k.EnablePoolIndicesOutput();
        }

        return k;
    }
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
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const pooling_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const pooling_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    Datatype GetAccumulatorType(const pooling_params& p) const;
    Datatype GetActivationType(const pooling_params& params) const;
    bool NeedsBoundaryCheck(const pooling_params& params) const;
    bool EnableRound(const pooling_params& params) const;
};
}  // namespace kernel_selector
