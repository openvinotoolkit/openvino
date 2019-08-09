// Copyright (c) 2019 Intel Corporation
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

#include "pooling_kernel_gpu_b_fs_yx_fsv4.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_b_fs_yx_fsv4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::bfyx);
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

PoolingKernelBase::DispatchData PoolingKerneGPU_b_fs_yx_fsv4::SetDefault(const pooling_params& params) const {
    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    runInfo.gws0 = params.output.X().v;  // X
    runInfo.gws1 = params.output.Y().v;  // Y
    // we got b_fs_yx_fsv4 format, we process 4 features per workitem
    runInfo.gws2 = (params.output.Feature().v * params.output.Batch().v) / 4;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    return runInfo;
}

JitConstants PoolingKerneGPU_b_fs_yx_fsv4::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, kd);

    const size_t in_x_pitch = 4;
    const size_t in_y_pitch = 4 * params.inputs[0].X().LogicalDimPadded();
    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));

    return jit;
}

KernelsData PoolingKerneGPU_b_fs_yx_fsv4::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector
