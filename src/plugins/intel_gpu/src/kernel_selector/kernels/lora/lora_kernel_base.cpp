// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

bool LoRAKernelBase::Validate(const Params& p) const {
    return p.GetType() == KernelType::LORA;
}

JitConstants LoRAKernelBase::GetJitConstants(const lora_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& state_dtype = params.inputs[2].GetDType();
    jit.Merge(MakeTypeJitConstants(state_dtype, "STATE"));

    auto acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));

    DimensionAccessHelperJit dims(params.inputs[3]);
    jit.AddConstant(MakeJitConstant("LORA_RANK", dims.f()));

    size_t lora_count = (params.inputs.size() - 2ul) / 3ul;
    jit.AddConstant(MakeJitConstant("LORA_COUNT", lora_count));

    return jit;
}

}  // namespace kernel_selector
