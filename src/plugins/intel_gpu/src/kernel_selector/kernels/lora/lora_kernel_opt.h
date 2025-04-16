// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lora_kernel_base.h"

namespace kernel_selector {

class LoRAKernelOpt : public LoRAKernelBase {
public:
    using Parent = LoRAKernelBase;
    LoRAKernelOpt() : LoRAKernelBase("lora_opt") {}
    virtual ~LoRAKernelOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    JitConstants GetJitConstants(const lora_params& params, size_t kernel_idx) const;
    CommonDispatchData SetDefault(const lora_params& params, size_t kernel_idx) const;
};

}  // namespace kernel_selector
