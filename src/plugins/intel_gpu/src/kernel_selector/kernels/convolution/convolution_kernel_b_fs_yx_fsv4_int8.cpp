// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_b_fs_yx_fsv4_int8.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
constexpr size_t sub_group_size = 16;

ParamsKey ConvolutionKernel_b_fs_yx_fsv4_int8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv4_int8::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv4_int8::SetDefault(const convolution_params& cp, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    dispatchData.gws[0] = CeilDiv(cp.outputs[0].X().v, sub_group_size) / 2;
    dispatchData.gws[1] = cp.outputs[0].Y().v;
    dispatchData.gws[2] = sub_group_size;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = sub_group_size;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv4_int8::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    if (p.outputs[0].X().v > 512 && p.filterSize.x == 5 && p.filterSize.y == 5)
        return FORCE_PRIORITY_2;
    else
        return FORCE_PRIORITY_9;
}

bool ConvolutionKernel_b_fs_yx_fsv4_int8::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p) || !ConvolutionCheckInput(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);
    if (params.inputs[0].X().v % 64)
        return false;

    bool bFilterSize = (params.filterSize.x == 5 && params.filterSize.y == 5) ||
                       (params.filterSize.x == 3 && params.filterSize.y == 3 && (params.inputs[0].Feature().v % 4) == 0) ||
                       (params.filterSize.x == 1 && params.filterSize.y == 1);

    bool bStride = (params.stride.x == 1 && params.stride.y == 1);

    if (!bFilterSize || !bStride || (params.outputs[0].Feature().v % 4) != 0 || (params.outputs[0].Batch().v != 1)) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv4_int8::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[2]));

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf0 = { "_0", {"batch", "FILTER_OFM_MAX * iter + ofm + 0", "idy", "idx"}, "res0", input_dt, 1 };
        FusedOpsConfiguration conf1 = { "_1", {"batch", "FILTER_OFM_MAX * iter + ofm + 1", "idy", "idx"}, "res1", input_dt, 1 };
        FusedOpsConfiguration conf2 = { "_2", {"batch", "FILTER_OFM_MAX * iter + ofm + 2", "idy", "idx"}, "res2", input_dt, 1 };
        FusedOpsConfiguration conf3 = { "_3", {"batch", "FILTER_OFM_MAX * iter + ofm + 3", "idy", "idx"}, "res3", input_dt, 1 };
        FusedOpsConfiguration conf4 = { "_4", {"batch", "FILTER_OFM_MAX * iter + ofm + 0", "idy", "idx"}, "res4", input_dt, 1 };
        FusedOpsConfiguration conf5 = { "_5", {"batch", "FILTER_OFM_MAX * iter + ofm + 1", "idy", "idx"}, "res5", input_dt, 1 };
        FusedOpsConfiguration conf6 = { "_6", {"batch", "FILTER_OFM_MAX * iter + ofm + 2", "idy", "idx"}, "res6", input_dt, 1 };
        FusedOpsConfiguration conf7 = { "_7", {"batch", "FILTER_OFM_MAX * iter + ofm + 3", "idy", "idx"}, "res7", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf0, conf1, conf2, conf3, conf4, conf5, conf6, conf7 }));
    }

    return jit;
}

KernelsData ConvolutionKernel_b_fs_yx_fsv4_int8::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

}  // namespace kernel_selector
