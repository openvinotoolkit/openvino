// Copyright (c) 2020 Intel Corporation
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


#include "pooling_kernel_gpu_byxf_af32.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_byxf_af32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    return k;
}

PoolingKernelBase::DispatchData PoolingKerneGPU_byxf_af32::SetDefault(const pooling_params& params) const {
    constexpr int simdSize = 8;

    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    runInfo.gws0 = params.output.X().v;
    runInfo.gws1 = params.output.Y().v;
    // we got byxf_af32 format, so if we process 4 features per workitem, that means we process 32 per simd, so divide
    // by 4 and we end up with 8
    runInfo.gws2 = (RoundUp(params.output.Feature().v, 32) * params.output.Batch().v) / 4;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = simdSize;

    return runInfo;
}

JitConstants PoolingKerneGPU_byxf_af32::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    JitConstants jit = PoolingKernelBase::GetJitConstants(params, kd);

    jit.AddConstant(MakeJitConstant("AS_INPUT_TYPE(val)", "as_" + toCLType(params.inputs[0].GetDType()) + "4(val)"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"",
                                     {"b", "f", "y", "x"},
                                     "fused_pool_result",
                                     input_dt,
                                     4,
                                     LoadType::LT_UNALIGNED,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::FEATURE};
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}


KernelsData PoolingKerneGPU_byxf_af32::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector
