// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
bool RoPEKernelBase::Validate(const Params& p) const {
    return KernelBaseOpenCL::Validate(p);
}

JitConstants RoPEKernelBase::GetJitConstants(const rope_params& params, RoPEKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("HEAD_SIZE", params.head_size));
    jit.AddConstant(MakeJitConstant("ROTARY_NDIMS", params.rotary_ndims));
    jit.AddConstant(MakeJitConstant("HALF_ROTARY_NDIMS", params.rotary_ndims / 2));
    jit.AddConstant(MakeJitConstant("HEAD_COUNT", params.head_cnt));

    if (params.head_size > params.rotary_ndims) {
        jit.AddConstant(MakeJitConstant("ENABLE_IO_COPY", true));
    }

    if (params.gather_rank > 0) {
        jit.AddConstant(MakeJitConstant("ENABLE_GATHER", true));
        jit.AddConstant(MakeJitConstant("GATHER_RANK", params.gather_rank));
    }

    if (params.slice_stop > params.slice_start) {
        jit.AddConstant(MakeJitConstant("ENABLE_SLICE", true));

        auto f = toCodeString(params.inputs[0].Feature(), 1);
        auto x = toCodeString(params.inputs[0].X(), 2);
        auto y = toCodeString(params.inputs[0].Y(), 3);

        auto sliced_val = toCodeString(params.slice_stop - params.slice_start);
        auto sliced_x = params.axis == 3 ? sliced_val : x;
        auto sliced_y = params.axis == 2 ? sliced_val : y;

        jit.AddConstant(MakeJitConstant("SLICED_INPUT0_X_PITCH", 1));
        jit.AddConstant(MakeJitConstant("SLICED_INPUT0_Y_PITCH", sliced_x));
        jit.AddConstant(MakeJitConstant("SLICED_INPUT0_FEATURE_PITCH", sliced_x + "*" + sliced_y));
        jit.AddConstant(MakeJitConstant("SLICED_INPUT0_BATCH_PITCH", sliced_x + "*" + sliced_y + "*" + f));
        jit.AddConstant(MakeJitConstant("SLICED_INPUT0_OFFSET", 0));
        jit.AddConstant(MakeJitConstant("SLICED_FROM_START", toCodeString(params.slice_start)));

        if (params.axis == 2) {
            jit.AddConstant(MakeJitConstant("SLICED_FROM_END", "(" + y + "-" + toCodeString(params.slice_stop) + ")"));
        } else if (params.axis == 3) {
            jit.AddConstant(MakeJitConstant("SLICED_FROM_END", "(" + x + "-" + toCodeString(params.slice_stop) + ")"));
        } else {
            OPENVINO_THROW("[GPU] Invalid axis value for RoPE operation");
        }
    }

    if (params.transposed_input) {
        jit.AddConstant(MakeJitConstant("ENABLE_TRANSPOSE", true));
        jit.AddConstant(MakeJitConstant("TRANSPOSED_INPUT0_OFFSET", 0));
        jit.AddConstant(MakeJitConstant("TRANSPOSED_INPUT0_X_PITCH", 1));
        jit.AddConstant(MakeJitConstant("TRANSPOSED_INPUT0_Y_PITCH", "INPUT0_FEATURE_PITCH"));
        jit.AddConstant(MakeJitConstant("TRANSPOSED_INPUT0_FEATURE_PITCH", "INPUT0_Y_PITCH"));
        jit.AddConstant(MakeJitConstant("TRANSPOSED_INPUT0_BATCH_PITCH", "INPUT0_BATCH_PITCH"));
    }

    if (params.is_qwen) {
        jit.AddConstant(MakeJitConstant("QWEN", true));
    } else if (params.is_chatglm) {
        jit.AddConstant(MakeJitConstant("CHATGLM", true));
    } else {
        jit.AddConstant(MakeJitConstant("RotateHalf", true));
    }

    return jit;
}

RoPEKernelBase::DispatchData RoPEKernelBase::SetDefault(const rope_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;
    if (params.is_chatglm || params.is_qwen) {
        dims_by_gws = {{ Tensor::DataChannelName::BATCH }, { Tensor::DataChannelName::FEATURE },
                       { Tensor::DataChannelName::Y, Tensor::DataChannelName::X }};
        dispatchData.gws = {input.Batch().v,
                            input.Feature().v,
                            params.head_cnt * std::max(params.rotary_ndims / 2ul, params.head_size - params.rotary_ndims)};
    } else {
        dims_by_gws = {{ Tensor::DataChannelName::BATCH }, { Tensor::DataChannelName::Y },
                       { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::X }};
        dispatchData.gws = {input.Batch().v,
                            input.Y().v,
                            input.Feature().v * params.rotary_ndims / 2ul};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, input.GetLayout(), output.GetLayout(), dims_by_gws);

    return dispatchData;
}

void RoPEKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const rope_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData RoPEKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::ROPE);

    if (!Validate(params))
        return {};

    const rope_params& orgParams = static_cast<const rope_params&>(params);
    auto dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<rope_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     static_cast<int>(orgParams.num_of_inputs),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.outputs[0].is_dynamic());

    return {kd};
}

}  // namespace kernel_selector
