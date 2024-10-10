// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_quantize_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct dynamic_quantize_params : public base_params {
    dynamic_quantize_params() : base_params(KernelType::DYNAMIC_QUANTIZE) {}
    size_t fc_ifm_size = 0;
};

class DynamicQuantizeKernelRef : public KernelBaseOpenCL {
public:
    DynamicQuantizeKernelRef() : KernelBaseOpenCL("dynamic_quantize_gpu_ref") {}
    virtual ~DynamicQuantizeKernelRef() {}

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
