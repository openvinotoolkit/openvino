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

#include "pooling_kernel_gpu_b_fs_yx_fsv16_imad.h"
#include "kernel_selector_utils.h"

#define FEATURE_SLICE_SIZE 16

namespace kernel_selector {
ParamsKey PoolingKernelGPU_b_fs_yx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
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

PoolingKernelBase::DispatchData PoolingKernelGPU_b_fs_yx_fsv16_imad::SetDefault(const pooling_params& params) const {
    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    const auto& out = params.output;
    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    runInfo.gws0 = x;
    runInfo.gws1 = y;
    // we got b_fs_yx_fsv16 format, we process 16 features per workitem
    runInfo.gws2 = CeilDiv(f, FEATURE_SLICE_SIZE) * params.output.Batch().v;

    auto local = GetOptimalLocalWorkGroupSizes({ runInfo.gws0, runInfo.gws1, runInfo.gws2 }, params.engineInfo);

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants PoolingKernelGPU_b_fs_yx_fsv16_imad::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, kd);

    const size_t in_x_pitch = FEATURE_SLICE_SIZE;
    const size_t in_y_pitch = FEATURE_SLICE_SIZE * params.inputs[0].X().LogicalDimPadded();
    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));

    if (!params.fused_ops.empty()) {
        auto input_dt = EnableRound(params) ? Datatype::INT32 : GetActivationType(params);
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "pool_result[i]", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData PoolingKernelGPU_b_fs_yx_fsv16_imad::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}

bool PoolingKernelGPU_b_fs_yx_fsv16_imad::Validate(const Params& params, const optional_params& options) const {
    if (!PoolingKernelBase::Validate(params, options)) {
        return false;
    }
    auto p = dynamic_cast<const pooling_params&>(params);

    if (p.inputs[0].Feature().v % FEATURE_SLICE_SIZE != 0)
        return false;

    return true;
}
}  // namespace kernel_selector