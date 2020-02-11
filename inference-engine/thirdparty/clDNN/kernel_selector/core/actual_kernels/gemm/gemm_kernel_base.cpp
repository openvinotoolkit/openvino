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

#include "gemm_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants GemmKernelBase::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("ALPHA", params.alpha),
        MakeJitConstant("BETA", params.beta),
        MakeJitConstant("TRANSPOSE_INPUT0", params.transpose_input0),
        MakeJitConstant("TRANSPOSE_INPUT1", params.transpose_input1),
        MakeJitConstant("QUANTIZATION_TERM", params.quantization != QuantizationType::NONE),
    });

    return jit;
}

GemmKernelBase::DispatchData GemmKernelBase::SetDefault(const gemm_params& params) const {
    const auto& output = params.output;

    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
    std::vector<size_t> global = { output.X().v, output.Y().v, total_batches };

    const auto& local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData GemmKernelBase::GetCommonKernelsData(const Params& params,
                                                 const optional_params& options,
                                                 float estimated_time) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto run_info = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    uint32_t fused_deps_total = 0;
    for (auto& fused_dep : prim_params.fused_ops) {
        for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
            fused_deps_total++;
        }
    }

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     run_info,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     fused_deps_total);

    k_data.estimatedTime = estimated_time;

    return {k_data};
}

JitConstants GemmKernelBase::GetFusedPrimitivesJitConstants(const gemm_params&, const DispatchData&) const {
    return {};
}

bool GemmKernelBase::Validate(const Params& p, const optional_params&) const {
    const gemm_params& params = static_cast<const gemm_params&>(p);

    if (params.GetType() != KernelType::GEMM) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype GemmKernelBase::GetActivationType(const gemm_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::F32;

    return GetUnitType(params);
}

}  // namespace kernel_selector
