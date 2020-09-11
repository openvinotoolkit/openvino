/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "gemm_kernel_ref.h"

namespace kernel_selector {
ParamsKey GemmKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableTensorPitches();
    k.EnableQuantization(QuantizationType::SYMMETRIC);

    return k;
}

JitConstants GemmKernelRef::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    if (params.quantization != QuantizationType::NONE) {
        jit.Merge(MakeTypeJitConstants(Datatype::INT32, "ACCUMULATOR"));
        jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACTIVATION"));
    } else {
        jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACCUMULATOR"));
        jit.Merge(MakeTypeJitConstants(Datatype::F32, "ACTIVATION"));
    }

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = { "", {"b", "f", "y", "x"}, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData GemmKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
}

bool GemmKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    // int8 validation
    const auto& gmm_params = static_cast<const gemm_params&>(params);
    auto input_type = gmm_params.inputs[0].GetDType();
    auto input2_type = gmm_params.inputs[1].GetDType();
    auto output_type = gmm_params.output.GetDType();

    // int8/uint8 inputs (quantization case) require additional checks
    // require some additional checks.
    if ((input_type != Datatype::UINT8 && input_type != Datatype::INT8) &&
        (input2_type != Datatype::UINT8 && input2_type != Datatype::INT8) &&
        (output_type != Datatype::UINT8 && output_type != Datatype::INT8))
        return true;

    bool is_quantization = (input_type == Datatype::INT8 || input_type == Datatype::UINT8) &&
        (input2_type == Datatype::INT8 || input2_type == Datatype::UINT8) &&
        (output_type == Datatype::INT8 || output_type == Datatype::UINT8 ||
            output_type == Datatype::F32 || output_type == Datatype::F16);

    bool has_fused_op = (input_type == Datatype::F32 || input_type == Datatype::F16) &&
        (input2_type == Datatype::F32 || input2_type == Datatype::F16) &&
        !gmm_params.fused_ops.empty() &&
        (output_type == Datatype::INT8 || output_type == Datatype::UINT8);

    if (!is_quantization && !has_fused_op)
        return false;

    return true;
}
}  // namespace kernel_selector
