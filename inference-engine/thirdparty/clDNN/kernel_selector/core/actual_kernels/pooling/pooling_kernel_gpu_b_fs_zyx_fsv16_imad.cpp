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

#include "pooling_kernel_gpu_b_fs_zyx_fsv16_imad.h"
#include "kernel_selector_utils.h"

#define FEATURE_SLICE_SIZE 16

namespace kernel_selector {
ParamsKey PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
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

PoolingKernelBase::DispatchData PoolingKernelGPU_b_fs_zyx_fsv16_imad::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    const auto& out = params.output;
    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    if (IsGlobalPooling(params)) {
        y = params.inputs[0].Y().v;
        z = params.inputs[0].Z().v;
        dispatchData.gws[0] = b;
        dispatchData.gws[1] = Align(std::min(y * z, params.engineInfo.maxWorkGroupSize), FEATURE_SLICE_SIZE);
        // we got b_fs_yx_fsv16 format, we process 16 features per workitem
        dispatchData.gws[2] = CeilDiv(f, FEATURE_SLICE_SIZE);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = dispatchData.gws[1];
        dispatchData.lws[2] = 1;
    } else {
        dispatchData.gws[0] = x;
        dispatchData.gws[1] = y * z;
        // we got b_fs_yx_fsv16 format, we process 16 features per workitem
        dispatchData.gws[2] = CeilDiv(f, FEATURE_SLICE_SIZE) * b;

        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }
    return dispatchData;
}

JitConstants PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);

    const size_t in_x_pitch = FEATURE_SLICE_SIZE;
    const size_t in_y_pitch = FEATURE_SLICE_SIZE * params.inputs[0].X().LogicalDimPadded();
    const size_t in_z_pitch = FEATURE_SLICE_SIZE * params.inputs[0].Y().LogicalDimPadded() * params.inputs[0].X().LogicalDimPadded();
    const size_t in_size_x = params.inputs[0].X().v;
    const size_t in_size_y = params.inputs[0].Y().v;
    const size_t in_size_z = params.inputs[0].Z().v;
    const auto y_load = CeilDiv(in_size_y, params.engineInfo.maxWorkGroupSize);
    const auto z_load = CeilDiv(in_size_z, std::max(params.engineInfo.maxWorkGroupSize / in_size_y, (size_t)1));
    jit.AddConstants({
        MakeJitConstant("LWS", dispatchData.lws[1]),
        MakeJitConstant("LWS_SIZE", std::min(in_size_z * in_size_y, dispatchData.lws[1])),
        MakeJitConstant("Y_LOAD", y_load),
        MakeJitConstant("Z_LOAD", z_load),
    });
    size_t maxUnrollSize = 256;
    size_t unrollX = 1;
    size_t unrollY = 1;
    size_t unrollZ = 1;
    if (in_size_x < maxUnrollSize)
        unrollX = in_size_x;
    else if (in_size_x % maxUnrollSize == 0)
        unrollX = maxUnrollSize;
    maxUnrollSize /= unrollX;
    if (in_size_y < maxUnrollSize)
        unrollY = in_size_y;
    else if (in_size_y % maxUnrollSize == 0)
        unrollY = maxUnrollSize;
    maxUnrollSize /= unrollY;
    if (in_size_z < maxUnrollSize)
        unrollZ = in_size_z;
    else if (in_size_z % maxUnrollSize == 0)
        unrollZ = maxUnrollSize;
    maxUnrollSize /= unrollZ;
    jit.AddConstant(MakeJitConstant("UNROLL_X", unrollX));
    jit.AddConstant(MakeJitConstant("UNROLL_Y", unrollY));
    jit.AddConstant(MakeJitConstant("UNROLL_Z", unrollZ));
    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
    jit.AddConstant(MakeJitConstant("IN_Z_PITCH", in_z_pitch));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    if (IsGlobalPooling(params))
        jit.AddConstant(MakeJitConstant("GLOBAL_POOLING", 1));

    if (!params.fused_ops.empty()) {
        auto input_dt = EnableRound(params) ? Datatype::INT32 : GetActivationType(params);
        FusedOpsConfiguration conf = {"", {"b", "(f+i)", "y", "x"}, "pool_result[i]", input_dt, 1};
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            conf = {"", {"b", "(f+i)", "z", "y", "x"}, "pool_result[i]", input_dt, 1 };
        }
        conf.SetLoopAxes({ Tensor::DataChannelName::FEATURE }, true);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

bool PoolingKernelGPU_b_fs_zyx_fsv16_imad::IsGlobalPooling(const pooling_params& params) const {
    return params.output.X().v == 1 && params.output.Y().v == 1 && params.output.Z().v == 1;
}

KernelsData PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

bool PoolingKernelGPU_b_fs_zyx_fsv16_imad::Validate(const Params& params, const optional_params& options) const {
    return PoolingKernelBase::Validate(params, options);
}
}  // namespace kernel_selector
