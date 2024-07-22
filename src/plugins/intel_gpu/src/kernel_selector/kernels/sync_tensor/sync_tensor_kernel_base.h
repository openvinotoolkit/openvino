// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
//#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// sync_tensor_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct sync_tensor_params : public base_params {
    sync_tensor_params() : base_params(KernelType::SYNC_TENSOR) {}

    SyncTensorDim dim = SyncTensorDim::FEATURE;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnableSyncTensorDim(dim);
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SyncTensorKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SyncTensorKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~SyncTensorKernelBase() {}

    struct DispatchData : public CommonDispatchData {
        size_t itemsNum;
        size_t leftovers;
        size_t dataSetsCount;
        size_t dataSetSize;
        size_t maxSlmSize;
        size_t normIndex;  // which dimension (from in-memory representation) is normalized, e.g. for bfyx and
                           // sync_tensor::normalize_f, it will be f's index == 2 (used only by naive kernel)
        size_t subgroupBlockSize;
    };

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const sync_tensor_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const sync_tensor_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    Datatype GetActivationType(const sync_tensor_params& params) const {
        if (params.inputs[0].GetDType() == Datatype::F16)
            return Datatype::F16;
        else
            return Datatype::F32;
    }
};
}  // namespace kernel_selector
