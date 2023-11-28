// Copyright (C) 2023 Intel Corporation
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

static std::size_t FinalOuotputBufferSize(const group_normalization_params &params) {
    const auto& output = params.outputs[0];
    return output.Batch().v * output.Feature().v * output.Z().v * output.Y().v * output.X().v * sizeof(float);
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
    case eCalcPow:
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

    // std::cout << "====" << params.layerID << ", id=" << id << ", groups=" << params.num_groups << ", maxWGsize=" << params.engineInfo.maxWorkGroupSize;
    // std::cout << ", ==gws";
    // for (auto g : dispatch_data.gws)
    //     std::cout << " " << g;
    // std::cout << ", ==lws";
    // for (auto l : dispatch_data.lws)
    //     std::cout << " " << l;
    // auto in0 = params.inputs[0];
    // std::cout << " , input0 bfxyz=" << in0.Batch().v << " "<< in0.Feature().v << " "<< in0.X().v << " "<< in0.Y().v << " "<< in0.Z().v << " ";
    // std::cout << " , output0 bfxyz=" << output.Batch().v << " "<< output.Feature().v << " "<< output.X().v << " "<< output.Y().v << " "<< output.Z().v << " ";
    // std::cout << std::endl;

    return dispatch_data;
}

JitConstants GroupNormalizationKernelRef::GetJitConstants(KernelId kernelId,
                                                          const group_normalization_params &params) const {
    auto jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("EPSILON", static_cast<float>(params.epsilon)));
    jit.AddConstant(MakeJitConstant("NUM_GROUPS", params.num_groups));
    switch (kernelId) {
    case eCalcMeanKernel:
        jit.AddConstant(MakeJitConstant("MEAN_KERNEL_ENABLED", true));
        break;
    case eCalcPow:
        jit.AddConstant(MakeJitConstant("CALC_POW_KERNEL_ENABLED", true));
        break;
    case eCalcStandardDeviationKernel:
        jit.AddConstant(MakeJitConstant("STANDARD_DEVIATION_KERNEL_ENABLED", true));
        break;
    //case eMergedNormalize: //kelvin test--------
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
    case eCalcPow: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        internalBufferSizes.push_back(FinalOuotputBufferSize(params));
        break;
    }
    case eCalcStandardDeviationKernel: {
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        internalBufferSizes.push_back(InternalBufferSize(params));
        break;
    }
    case eNormalize: {
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        break;
    }
    // case eMergedNormalize: {
    //     arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    //     arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    //     arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    //     arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    //     arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    //     arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    //     internalBufferSizes.push_back(InternalBufferSize(params));
    //     internalBufferSizes.push_back(InternalBufferSize(params));
    //     break;
    // }
    default:
        assert(false);
        break;
    }
}

KernelsData GroupNormalizationKernelRef::GetKernelsData(const Params &params, const optional_params &options) const {
    KernelData kd = KernelData::Default<group_normalization_params>(params, eKernelsNum);
    group_normalization_params& parameters = *static_cast<group_normalization_params*>(kd.params.get());

    kd.internalBufferDataType = Datatype::F32;
    //std::cout << "==========params.outputs[0].ElementSize: " << parameters.outputs[0].ElementSize() << std::endl;
    for (KernelId id = eCalcMeanKernel; id < eKernelsNum; ++id) {
    //{
        //KernelId id = eMergedNormalize;
        //auto& kernel = kd.kernels[0];
        auto& kernel = kd.kernels[id];
        const auto entryPoint = GetEntryPoint(kernelName, parameters.layerID, params, options, id);
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

} // namespace kernel_selector
