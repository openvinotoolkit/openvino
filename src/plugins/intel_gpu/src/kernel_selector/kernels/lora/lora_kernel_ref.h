// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lora_kernel_base.h"

namespace kernel_selector {

class LoRAKernelRef : public LoRAKernelBase {
public:
    using Parent = LoRAKernelBase;
    LoRAKernelRef() : LoRAKernelBase("lora_ref") {}
    virtual ~LoRAKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    JitConstants GetJitConstants(const lora_params& params, size_t kernel_idx) const;
    CommonDispatchData SetDefault(const lora_params& params, size_t kernel_idx) const;
};

}  // namespace kernel_selector
