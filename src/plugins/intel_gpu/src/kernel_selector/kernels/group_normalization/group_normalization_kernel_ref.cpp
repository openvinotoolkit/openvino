// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_kernel_ref.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey GroupNormalizationKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static std::size_t InternalBufferSize(const group_normalization_params &params) {
    const auto& output = params.outputs[0];
    return output.Batch().v * params.num_groups * sizeof(float);
}

static GroupNormalizationKernelRef::KernelId operator++(GroupNormalizationKernelRef::KernelId& id) {
    id = static_cast<GroupNormalizationKernelRef::KernelId>(static_cast<int>(id) + 1);
    return id;
}

GroupNormalizationKernelRef::DispatchData GroupNormalizationKernelRef::SetDefault(
    KernelId id, const group_normalization_params &params) const {
    DispatchData dispatch_data;
    auto& output = params.outputs[0];
    switch (id) {
    case eCalcMeanKernel:
    case eCalcStandardDeviationKernel: {
        auto maxWorkGroupSize = params.engineInfo.maxWorkGroupSize;
        dispatch_data.gws = std::vector<std::size_t>{
            output.Batch().v,
            static_cast<std::size_t>(params.num_groups),
            1
        };
        dispatch_data.lws = std::vector<std::size_t>{
            output.Batch().v * params.num_groups > maxWorkGroupSize ? maxWorkGroupSize / params.num_groups : output.Batch().v,
            static_cast<std::size_t>(params.num_groups),
            1};
        break;
    }
    case eNormalize: {
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = output.GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
            { Tensor::DataChannelName::BATCH },
            { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::Z  },
            { Tensor::DataChannelName::X, Tensor::DataChannelName::Y }};
        dispatch_data.gws = std::vector<std::size_t>{
            output.Batch().v,
            output.Feature().v * output.Z().v,
            output.X().v * output.Y().v};
        dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo,
            in_layout, out_layout, dims_by_gws);
        break;
    }
    default:
        assert(false);
        break;
    }
    return dispatch_data;
}

JitConstants GroupNormalizationKernelRef::GetJitConstants(KernelId kernelId,
                                                          const group_normalization_params &params) const {
    auto jit = GroupNormalizationKernelBase::GetJitConstants(params);

    switch (kernelId) {
    case eCalcMeanKernel:
        jit.AddConstant(MakeJitConstant("MEAN_KERNEL_ENABLED", true));
        break;
    case eCalcStandardDeviationKernel:
        jit.AddConstant(MakeJitConstant("STANDARD_DEVIATION_KERNEL_ENABLED", true));
        break;
    case eNormalize: {
        jit.AddConstant(MakeJitConstant("NORMALIZE_KERNEL_ENABLED", true));
        jit.AddConstant(MakeJitConstant("INPUT_INDICES_ORDER", "batch, feature, z, y, x"));
        if (!params.fused_ops.empty()) {
            FusedOpsConfiguration conf{
                "",
                params.outputs[0].Dimentions() == 5 ? std::vector<std::string>{"batch", "feature", "z", "y", "x"} :
                    std::vector<std::string>{"batch", "feature", "y", "x"},
                "res",
                params.outputs[0].GetDType()
            };
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
        break;
    }
    default:
        assert(false);
        break;
    }
    return jit;
}

void GroupNormalizationKernelRef::SetKernelArguments(const group_normalization_params& params,
                                                     KernelId kernelId,
                                                     cldnn::arguments_desc& arguments,
                                                     std::vector<std::size_t>& internalBufferSizes) {
    switch (kernelId) {
    case eCalcMeanKernel: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        internalBufferSizes.push_back(InternalBufferSize(params));
        break;
    }
    case eCalcStandardDeviationKernel: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        internalBufferSizes.push_back(InternalBufferSize(params));
        break;
    }
    case eNormalize: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        break;
    }
    default:
        assert(false);
        break;
    }
}

KernelsData GroupNormalizationKernelRef::GetKernelsData(const Params &params) const {
    const group_normalization_params& parameters = static_cast<const group_normalization_params&>(params);
    KernelData kd = KernelData::Default<group_normalization_params>(params, eKernelsNum);
    kd.internalBufferDataType = Datatype::F32;
    for (KernelId id = eCalcMeanKernel; id < eKernelsNum; ++id) {
        auto& kernel = kd.kernels[id];
        const auto entryPoint = GetEntryPoint(kernelName, parameters.layerID, params, id);
        auto jitConstants = GetJitConstants(id, parameters);
        const auto jit = CreateJit(kernelName, jitConstants, entryPoint);
        const auto dispatchData = SetDefault(id, parameters);
        FillCLKernelData(kernel,
                         dispatchData,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entryPoint,
                         "",
                         false,
                         false,
                         0,
                         0,
                         0);
        SetKernelArguments(parameters, id, kernel.params.arguments, kd.internalBufferSizes);
    }
    return {kd};
}

KernelsPriority GroupNormalizationKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
} // namespace kernel_selector
