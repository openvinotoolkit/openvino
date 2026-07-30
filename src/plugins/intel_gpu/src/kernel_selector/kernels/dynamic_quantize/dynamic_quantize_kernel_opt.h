// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "kernel_base_opencl.h"
#include "dynamic_quantize_kernel_ref.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_quantize_params
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DynamicQuantizeKernelOpt : public KernelBaseOpenCL {
public:
    explicit DynamicQuantizeKernelOpt(const std::string& kernel_name = "dynamic_quantize_gpu_opt") : KernelBaseOpenCL(kernel_name) {}
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

class DynamicQuantizeKernelOptOrgRefToBeReverted : public DynamicQuantizeKernelOpt {
public:
    DynamicQuantizeKernelOptOrgRefToBeReverted() : DynamicQuantizeKernelOpt("dynamic_quantize_gpu_opt_org_ref_to_be_reverted") {}
};
}  // namespace kernel_selector
