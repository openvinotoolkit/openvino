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

    const auto& lora_rank_dim = params.inputs[3].Feature();
    std::string lora_rank = lora_rank_dim.is_dynamic ? toShapeInfoString(3, 1)
                                                     : toCodeString(lora_rank_dim.v);
    jit.AddConstant(MakeJitConstant("LORA_RANK", lora_rank));

    size_t lora_count = (params.inputs.size() - 2ul) / 3ul;
    jit.AddConstant(MakeJitConstant("LORA_COUNT", lora_count));

    return jit;
}

}  // namespace kernel_selector
