// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

DeviceFeaturesKey PoolingKernelGPU_b_fs_zyx_fsv16_imad::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

PoolingKernelBase::DispatchData PoolingKernelGPU_b_fs_zyx_fsv16_imad::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = params.outputs[0];
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

        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
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
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            conf = {"", {"b", "(f+i)", "z", "y", "x"}, "pool_result[i]", input_dt, 1 };
        }
        conf.SetLoopAxes({ Tensor::DataChannelName::FEATURE }, true);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

bool PoolingKernelGPU_b_fs_zyx_fsv16_imad::IsGlobalPooling(const pooling_params& params) const {
    return params.outputs[0].X().v == 1 && params.outputs[0].Y().v == 1 && params.outputs[0].Z().v == 1;
}

KernelsData PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority PoolingKernelGPU_b_fs_zyx_fsv16_imad::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

bool PoolingKernelGPU_b_fs_zyx_fsv16_imad::Validate(const Params& params) const {
    return PoolingKernelBase::Validate(params);
}
}  // namespace kernel_selector
