// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// lora_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct lora_params : public base_params {
    lora_params() : base_params(KernelType::LORA) {}
};

struct lora_fuse_params : fuse_params {
    lora_fuse_params() : fuse_params(KernelType::LORA) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LoRAKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LoRAKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~LoRAKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const lora_params& params) const;
};
}  // namespace kernel_selector
