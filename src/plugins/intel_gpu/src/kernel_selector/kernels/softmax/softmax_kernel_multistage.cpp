// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_multistage.hpp"

namespace kernel_selector {

ParamsKey SoftmaxKernel_multistage::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableSoftmaxDim(SoftmaxDim::X);  // in case that it can be flatten
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    return k;
}

KernelsData SoftmaxKernel_multistage::GetKernelsData(const Params &params,
                                                     const optional_params &options) const {
    if (!Validate(params, options))
        return {};
    KernelData kd = KernelData::Default<softmax_params>(params, 3);
    const softmax_params& softmaxParams = static_cast<const softmax_params&>(params);
    const int bytesPerElement = softmaxParams.inputs[0].ElementSize();
    size_t entryPointId = 0;
    {
        auto dispatchData = SetDefault(softmaxParams);
        auto jitConstants = GetJitConstants(softmaxParams, dispatchData);
        jitConstants.AddConstant(MakeJitConstant("MAX_REDUCE_KERNEL", 1));
        auto entryPoint = GetEntryPoint(kernelName, softmaxParams.layerID, params, options, entryPointId++);
        auto jit = CreateJit(kernelName, jitConstants, entryPoint);
        auto& kernel = kd.kernels[0];
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
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kd.internalBufferSizes.push_back(dispatchData.gws[0] * (dispatchData.gws[1] / (kSubGroupSize * kElementsPerThread)) * bytesPerElement);
    }
    {
        auto dispatchData = SetDefault(softmaxParams);
        auto jitConstants = GetJitConstants(softmaxParams, dispatchData);
        jitConstants.AddConstant(MakeJitConstant("ADD_REDUCE_KERNEL", 1));
        auto entryPoint = GetEntryPoint(kernelName, softmaxParams.layerID, params, options, entryPointId++);
        auto jit = CreateJit(kernelName, jitConstants, entryPoint);
        auto& kernel = kd.kernels[1];
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
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBufferSizes.push_back(dispatchData.gws[0] * (dispatchData.gws[1] / (kSubGroupSize * kElementsPerThread)) * bytesPerElement);
        kd.internalBufferSizes.push_back(dispatchData.gws[0] * bytesPerElement);
    }
    {
        auto dispatchData = SetDefault(softmaxParams);
        auto jitConstants = GetJitConstants(softmaxParams, dispatchData);
        jitConstants.AddConstant(MakeJitConstant("SOFTMAX_KERNEL", 1));
        auto entryPoint = GetEntryPoint(kernelName, softmaxParams.layerID, params, options, entryPointId++);
        auto jit = CreateJit(kernelName, jitConstants, entryPoint);
        auto& kernel = kd.kernels[2];
        FillCLKernelData(kernel,
                         dispatchData,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entryPoint);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
    }
    kd.internalBufferDataType = softmaxParams.inputs[0].GetDType();
    return {kd};
}

size_t SoftmaxKernel_multistage::GetDataSetSize(const softmax_params &params) {
    const auto& input = params.inputs[0];

    if (input.X().v == 1 && params.dim == SoftmaxDim::Y && input.Feature().v > 1 && input.GetLayout() == DataLayout::bfyx) {
        return input.Y().v;
    } else {
        auto flatten_input = input.FlattenFeatureAndSpatials();
        return flatten_input.Feature().v;
    }
}

size_t SoftmaxKernel_multistage::GetDataSetCount(const softmax_params &params) {
    const auto& input = params.inputs[0];

    if (input.X().v == 1 && params.dim == SoftmaxDim::Y && input.Feature().v > 1 && input.GetLayout() == DataLayout::bfyx) {
        return input.Batch().v * input.Feature().v;
    } else {
        auto flatten_input = input.FlattenFeatureAndSpatials();
        return input.Batch().v;
    }
}

float SoftmaxKernel_multistage::GetKernelsPriority(const Params &params,
                                                      const optional_params &options) const {
    const softmax_params& p = static_cast<const softmax_params&>(params);
    return GetDataSetSize(p) >= 4096 ? FORCE_PRIORITY_5 : DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

DeviceFeaturesKey SoftmaxKernel_multistage::get_required_device_features_key(
    const Params &params, const optional_params& options) const {
    auto k = get_common_subgroups_device_features_key(params, options);
    k.requires_subgroup_reduce();
    return k;
}

bool SoftmaxKernel_multistage::Validate(const Params& params, const optional_params& options) const {
    if (!SoftmaxKernelBase::Validate(params, options))
        return false;

//    if (!params.activations.empty()) {
//        return false;
//    }

    const auto& input = static_cast<const softmax_params&>(params).inputs[0];
    if (input.GetLayout() == DataLayout::bf || input.GetLayout() == DataLayout::fb) {
        return true;
    }

    switch (static_cast<const softmax_params&>(params).dim) {
    case SoftmaxDim::X:
        return input.Y().v == 1 && input.Z().v == 1 && input.Feature().v == 1;
    case SoftmaxDim::Y:
        return input.X().v == 1 && input.Z().v == 1 && (input.Feature().v == 1 || input.GetLayout() == DataLayout::bfyx);
    case SoftmaxDim::Z:
        return input.X().v == 1 && input.Y().v == 1 && input.Feature().v == 1;
    case SoftmaxDim::FEATURE:
        return input.X().v == 1 && input.Y().v == 1 && input.Z().v == 1;
    default:
        return false;
    }

    const softmax_params& softmaxParams = static_cast<const softmax_params&>(params);
    return GetDataSetSize(softmaxParams) % (params.engineInfo.maxWorkGroupSize * kElementsPerThread) == 0;
}

JitConstants SoftmaxKernel_multistage::GetJitConstants(const softmax_params &params, DispatchData dispatchData) const {
    JitConstants jitConstants = SoftmaxKernelBase::GetJitConstants(params, dispatchData);
    jitConstants.AddConstants({MakeJitConstant("ELEMENTS_PER_THREAD", kElementsPerThread)});
    return jitConstants;
}

SoftmaxKernelBase::DispatchData SoftmaxKernel_multistage::SetDefault(const softmax_params &params) const {
    SoftmaxKernelBase::DispatchData dispatchData;
    dispatchData.dataSetsCount = GetDataSetCount(params);
    dispatchData.gws[0] = dispatchData.dataSetsCount;
    dispatchData.dataSetSize = GetDataSetSize(params);
    dispatchData.gws[1] = dispatchData.dataSetSize / kElementsPerThread;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = params.engineInfo.maxWorkGroupSize;
    dispatchData.lws[2] = 1;
    return dispatchData;
}

} // namespace kernel_selector
