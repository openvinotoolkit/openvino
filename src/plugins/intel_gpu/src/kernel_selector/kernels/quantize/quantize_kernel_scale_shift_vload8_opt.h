// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "quantize_kernel_base.h"

namespace kernel_selector {

class QuantizeKernelScaleShift_vload8 : public QuantizeKernelBase {
public:
    using Parent = QuantizeKernelBase;

    QuantizeKernelScaleShift_vload8() : QuantizeKernelBase("quantize_gpu_scale_shift_vload8_opt") {}
    virtual ~QuantizeKernelScaleShift_vload8() {}
    CommonDispatchData SetDefault(const quantize_params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const override;
};
}  // namespace kernel_selector
