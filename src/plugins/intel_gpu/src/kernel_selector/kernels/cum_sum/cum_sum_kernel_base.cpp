// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
Tensor::DataChannelName CumSumKernelBase::GetCumSumAxis(const cum_sum_params& params) const {
    switch (params.axis) {
        case CumSumAxis::X:
            return Tensor::DataChannelName::X;
        case CumSumAxis::Y:
            return Tensor::DataChannelName::Y;
        case CumSumAxis::Z:
            return Tensor::DataChannelName::Z;
        case CumSumAxis::W:
            return Tensor::DataChannelName::W;
        case CumSumAxis::FEATURE:
            return Tensor::DataChannelName::FEATURE;
        case CumSumAxis::BATCH:
            return Tensor::DataChannelName::BATCH;
        default:
            return Tensor::DataChannelName::BATCH;
    }
}

int32_t CumSumKernelBase::GetCumSumAxisIndex(const cum_sum_params& params) const {
    return DataTensor::Channelndex(params.outputs[0].GetLayout(), GetCumSumAxis(params));
}

size_t CumSumKernelBase::GetRealAxisIndex(const cum_sum_params& params) const {
    size_t index = params.outputs[0].Dimentions() - GetCumSumAxisIndex(params) - 1;
    if (params.outputs[0].Dimentions() == 6)
        return index;
    else if (params.outputs[0].Dimentions() == 5)
        return (index > 1) ? index + 1 : index;
    return (index > 1) ? index + 2 : index;
}

JitConstants CumSumKernelBase::GetJitConstants(const cum_sum_params& params, DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.exclusive)
        jit.AddConstant(MakeJitConstant("EXCLUSIVE", 1));
    if (params.reverse)
        jit.AddConstant(MakeJitConstant("REVERSE", 1));
    jit.AddConstant(MakeJitConstant("AXIS", GetRealAxisIndex(params)));

    return jit;
}

CumSumKernelBase::DispatchData CumSumKernelBase::SetDefault(const cum_sum_params& params) const {
    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::W, Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y, Tensor::DataChannelName::Z }};

    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v * params.outputs[0].W().v,
                         params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

void CumSumKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const cum_sum_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData CumSumKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<cum_sum_params>(params);
    cum_sum_params& newParams = *static_cast<cum_sum_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, 1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     newParams.is_shape_agnostic);

    return {kd};
}

bool CumSumKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::CUM_SUM) {
        return false;
    }

    auto& params = static_cast<const cum_sum_params&>(p);
    if (GetCumSumAxisIndex(params) == -1)
        return false;

    if (params.inputs.size() > 1 && params.inputs[1].GetDType() != Datatype::INT32 &&
        params.inputs[1].GetDType() != Datatype::UINT32)
        return false;

    return true;
}

Datatype CumSumKernelBase::GetActivationType(const cum_sum_params& params) const {
    if (params.outputs[0].GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}
}  // namespace kernel_selector
