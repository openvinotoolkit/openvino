// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ActivationKernelBase::DispatchData ActivationKernelBase::SetDefault(const activation_params& arg) const {
    const auto& out = arg.outputs[0];
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();

    DispatchData dispatchData;
    if (out_layout == DataLayout::yxfb) {
        dispatchData.gws = {out.Feature().v * out.Batch().v, out.X().v, out.Y().v};
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH},
                                                                         {Tensor::DataChannelName::X},
                                                                         {Tensor::DataChannelName::Y}};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
    } else if (out_layout == DataLayout::b_fs_yx_fsv16 || out_layout == DataLayout::b_fs_yx_fsv32) {
        dispatchData.gws = {Align(out.Feature().v, 16) * out.Batch().v, out.X().v, out.Y().v};
        dispatchData.lws = {16, 1, 1};
    } else if (out_layout == DataLayout::bs_fs_yx_bsv32_fsv16 || out_layout == DataLayout::bs_fs_yx_bsv32_fsv32) {
        dispatchData.gws = {out.X().v * out.Y().v, Align(out.Feature().v, 16), Align(out.Batch().v, 16)};
        dispatchData.lws = {1, 16, 16};
    } else {
        dispatchData.gws = {out.X().v, out.Y().v * out.Z().v * out.W().v * out.U().v * out.V().v, out.Feature().v * out.Batch().v};
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{Tensor::DataChannelName::X},
                                                                         {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                          Tensor::DataChannelName::U, Tensor::DataChannelName::V},
                                                                         {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

JitConstants ActivationKernelBase::GetJitConstants(const activation_params& params, DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& inputNlParams = params.inputActivationParams;

    jit.AddConstants({
        MakeJitConstant("PARAMS_NUM", GetActivationAdditionalParamsNumber(params.activations[0].function)),
    });

    if (!inputNlParams.empty()) {
        jit.AddConstants({
            MakeJitConstant("ADDITIONAL_PARAMS", inputNlParams[0]),
            MakeJitConstant("PARAMETERIZED", ""),
        });
    }

    if (params.has_dynamic_outputs()) {
        jit.AddConstant(MakeJitConstant("OUTPUT_BATCH_NUM_CONST", 0));
        jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_CONST", 0));
    } else {
        jit.AddConstant(MakeJitConstant("OUTPUT_BATCH_NUM_CONST", params.outputs[0].Batch().v));
        jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_CONST", params.outputs[0].Feature().v));
    }

    return jit;
}

bool ActivationKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ACTIVATION) {
        return false;
    }
    const activation_params& orgParams = static_cast<const activation_params&>(p);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

void ActivationKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const activation_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData ActivationKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<activation_params>(params);
    activation_params& newParams = *static_cast<activation_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1,
                     GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);

    if (!newParams.inputActivationParams.empty()) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SLOPE, 0});
    }

    return {kd};
}
}  // namespace kernel_selector
