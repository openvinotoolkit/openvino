// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rms_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rms_params : public base_params {
    rms_params() : base_params(KernelType::RMS) {}
    float epsilon = 0.0f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rms_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rms_optional_params : optional_params {
    rms_optional_params() : optional_params(KernelType::RMS) {}
};

class RMSKernelRef : public KernelBaseOpenCL {
public:
    RMSKernelRef() : KernelBaseOpenCL("rms_ref") {}
    virtual ~RMSKernelRef() {}

    struct DispatchData : public CommonDispatchData {
        size_t dataSize;
        size_t dataCount;
        size_t slmSize;

        DispatchData() : dataSize(0), dataCount(0), slmSize(0) {}
    };

    virtual JitConstants GetJitConstants(const rms_params& params, DispatchData DispatchData) const;
    virtual DispatchData SetDefault(const rms_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    Datatype GetAccumulatorType(const rms_params& params) const;
    bool Validate(const Params&, const optional_params&) const override;
};
}  // namespace kernel_selector
