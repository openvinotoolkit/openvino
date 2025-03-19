// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "tensor_type.h"
#include "concatenation_kernel_base.h"
#include <algorithm>
#include <vector>

namespace kernel_selector {
Tensor::DataChannelName ConcatenationKernelBase::GetConcatChannel(const concatenation_params& params) const {
    switch (params.axis) {
        case ConcatAxis::X:
            return Tensor::DataChannelName::X;
        case ConcatAxis::Y:
            return Tensor::DataChannelName::Y;
        case ConcatAxis::Z:
            return Tensor::DataChannelName::Z;
        case ConcatAxis::W:
            return Tensor::DataChannelName::W;
        case ConcatAxis::FEATURE:
            return Tensor::DataChannelName::FEATURE;
        case ConcatAxis::BATCH:
            return Tensor::DataChannelName::BATCH;
        default:
            return Tensor::DataChannelName::X;
    }
}

int32_t ConcatenationKernelBase::GetConcatChannelIndex(const concatenation_params& params) const {
    return DataTensor::Channelndex(params.outputs[0].GetLayout(), GetConcatChannel(params));
}

bool ConcatenationKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::CONCATENATION) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (GetConcatChannelIndex(params) == -1) {
        return false;
    }

    return true;
}

JitConstants ConcatenationKernelBase::GetJitConstants(const concatenation_params& params) const {
    auto& inputs = params.original_input_layouts;
    bool is_dynamic = std::any_of(inputs.begin(), inputs.end(), [](const DataTensor& t) { return t.is_dynamic(); }) ||
                      std::any_of(params.outputs.begin(), params.outputs.end(), [](const DataTensor& t) { return t.is_dynamic(); });
    JitConstants jit = MakeBaseParamsJitConstants(params, !is_dynamic);

    jit.AddConstants({
        MakeJitConstant("CONCAT_" + toString(params.axis), 1),
    });

    if (is_dynamic) {
        jit.AddConstant(MakeJitConstant("INPUT0", params.inputs[0]));
        jit.AddConstant(MakeJitConstant("OUTPUT", params.outputs[0]));

        jit.AddConstant(MakeJitConstant("IS_DYNAMIC", 1));
        jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", "__global const int* shape_info,"));
        jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", "shape_info,"));
    }

    jit.AddConstant(MakeJitConstant("CONCAT_AXIS_INDEX", GetConcatChannelIndex(params)));
    return jit;
}

ConcatenationKernelBase::DispatchData ConcatenationKernelBase::SetDefault(const concatenation_params& params) const {
    DispatchData dispatchData;

    const auto& dims = params.inputs[0].GetDims();
    auto layout = params.inputs[0].GetLayout();

    std::vector<int> idx = { DataTensor::Channelndex(layout, Tensor::DataChannelName::BATCH),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::FEATURE),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::Y),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::X) };

    // Determine global work sizes.
    dispatchData.gws[0] = idx[2] != -1 ? dims[idx[2]].v : 1;  // Y
    dispatchData.gws[1] = idx[1] != -1 ? dims[idx[1]].v : 1;  // F
    dispatchData.gws[2] = idx[0] != -1 ? dims[idx[0]].v : 1;  // B

    dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
        --dispatchData.lws[0];
    }

    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;
    return dispatchData;
}

void ConcatenationKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const concatenation_params&>(params);
        uint32_t lastOffset = 0;
        for (size_t i = 0; i < prim_params.inputs.size(); i++) {
            size_t ifm_offset = 0;

            const auto& input = prim_params.inputs[i];
            auto newParams = prim_params;
            newParams.inputs.resize(1);
            newParams.inputs[0] = input;
            size_t ifm = input.Feature().v;
            newParams.isAligned = ifm_offset % GetAlignment(newParams) == 0;
            newParams.misalignment = ifm_offset % GetAlignment(newParams);
            ifm_offset += ifm;

            auto& kernel = kd.kernels[i];
            DispatchData dispatchData = SetDefault(newParams);
            kernel.params.workGroups.global = dispatchData.gws;
            kernel.params.workGroups.local = dispatchData.lws;
            kernel.skip_execution = KernelData::SkipKernelExecution(newParams);

            ScalarDescriptor s;
            s.t = ScalarDescriptor::Types::UINT32;
            s.v.u32 = lastOffset;
            kernel.params.scalars.resize(1);
            kernel.params.scalars[0] = s;

            auto concatChannelIndex = DataTensor::Channelndex(input.GetLayout(), GetConcatChannel(prim_params));
            OPENVINO_ASSERT(concatChannelIndex >= 0, "concatChannelIndex shouldn't be negative");
            lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
        }
    };
}

KernelsData ConcatenationKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);
    KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    bool is_dynamic = orgParams.has_dynamic_tensors();
    uint32_t lastOffset = 0;
    size_t ifm_offset = 0;
    for (size_t i = 0; i < orgParams.inputs.size(); i++) {
        const auto& input = orgParams.inputs[i];
        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;
        size_t ifm = input.Feature().v;
        newParams.isAligned = ifm_offset % GetAlignment(newParams) == 0;
        newParams.misalignment = ifm_offset % GetAlignment(newParams);
        ifm_offset += ifm;

        newParams.kernel_split_id = i;
        newParams.original_input_layouts = orgParams.inputs;

        auto& kernel = kd.kernels[i];
        DispatchData dispatchData = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params, i);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local = dispatchData.lws;
        kernel.skip_execution = KernelData::SkipKernelExecution(newParams);
        if (is_dynamic) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t) i});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        ScalarDescriptor s;
        s.t = ScalarDescriptor::Types::UINT32;
        s.v.u32 = lastOffset;
        kernel.params.scalars.push_back(s);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        size_t concatChannelIndex = (size_t)DataTensor::Channelndex(orgParams.inputs[i].GetLayout(), GetConcatChannel(orgParams));
        lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
    }

    return {kd};
}
}  // namespace kernel_selector
