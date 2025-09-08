// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "dynamic_quantize_kernel_ref.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_quantize_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DynamicQuantizeKernelOpt : public KernelBaseOpenCL {
public:
    DynamicQuantizeKernelOpt() : KernelBaseOpenCL("dynamic_quantize_gpu_opt") {}
    virtual ~DynamicQuantizeKernelOpt() {}

    virtual JitConstants GetJitConstants(const dynamic_quantize_params& params) const;
    virtual CommonDispatchData SetDefault(const dynamic_quantize_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    Datatype GetAccumulatorType(const dynamic_quantize_params& params) const;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params&) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
