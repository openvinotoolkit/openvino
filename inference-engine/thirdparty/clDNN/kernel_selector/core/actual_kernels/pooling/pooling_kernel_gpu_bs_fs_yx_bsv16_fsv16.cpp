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

#include "pooling_kernel_gpu_bs_fs_yx_bsv16_fsv16.h"
#include "kernel_selector_utils.h"

//
// Kernel specific constants
//
#define SIMD_SIZE 16

namespace kernel_selector {
ParamsKey Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);

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

PoolingKernelBase::DispatchData Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    runInfo.gws0 = params.output.Feature().v/16;
    runInfo.gws1 = params.output.X().v * params.output.Y().v;
    runInfo.gws2 = params.output.Batch().v;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = SIMD_SIZE;
    runInfo.efficiency = FORCE_PRIORITY_1;

    return runInfo;
}

JitConstants Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, kd);

    if (!params.fused_ops.empty()) {
        auto input_dt = EnableRound(params) ? Datatype::INT32 : GetActivationType(params);
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "pool_result[i]", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}

bool Pooling_kernel_gpu_bs_fs_yx_bsv_16_fsv16::Validate(const Params& params, const optional_params& options) const {
    if (!PoolingKernelBase::Validate(params, options)) {
        return false;
    }
    auto p = dynamic_cast<const pooling_params&>(params);

    if (p.inputs[0].Feature().v % 16 != 0)
        return false;

    if (p.inputs[0].Batch().v % 16 != 0)
        return false;

    return true;
}
}  // namespace kernel_selector
