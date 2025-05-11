// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>

namespace kernel_selector {

ParamsKey AdaptivePoolingRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();

    return k;
}

KernelsPriority AdaptivePoolingRef::GetKernelsPriority(const Params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool AdaptivePoolingRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ADAPTIVE_POOLING) {
        return false;
    }

    const auto& params = dynamic_cast<const adaptive_pooling_params&>(p);
    const auto& inputs = params.inputs;

    if (params.mode == PoolType::MAX && params.outputs_num != 2) {
        return false;
    }

    const auto input_dims = inputs[0].Dimentions();
    if (input_dims < 2 || input_dims > 5) {
        return false;
    }

    return true;
}

namespace {
AdaptivePoolingRef::DispatchData SetDefault(const adaptive_pooling_params& params) {
    AdaptivePoolingRef::DispatchData dispatch_data;
    const auto& output = params.outputs[0];

    dispatch_data.gws[0] = output.X().v;
    dispatch_data.gws[1] = output.Y().v * output.Z().v;
    dispatch_data.gws[2] = output.Batch().v * output.Feature().v;

    dispatch_data.lws[0] = 1;
    dispatch_data.lws[1] = 1;
    dispatch_data.lws[2] = 1;

    return dispatch_data;
}
}  // namespace

KernelsData AdaptivePoolingRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<adaptive_pooling_params>(params);
    const adaptive_pooling_params& new_params = static_cast<const adaptive_pooling_params&>(params);

    const auto dispatchData = SetDefault(new_params);
    const auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);

    auto cldnn_jit = MakeBaseParamsJitConstants(new_params);

    cldnn_jit.AddConstant(MakeJitConstant(toString(new_params.mode) + "_POOLING", 1));

    const auto accumulator_type = new_params.inputs[0].GetDType();
    cldnn_jit.Merge(MakeTypeJitConstants(accumulator_type, "ACCUMULATOR"));

    const auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    KernelBase::CheckDispatchData(kernelName, dispatchData, params.engineInfo.maxWorkGroupSize);
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local  = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo);

    auto& arguments = kernel.params.arguments;
    arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});     // input data
    arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});    // output
    if (new_params.mode == PoolType::MAX) {
        arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});     // second output
    }

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kd));
    return kernelsData;
}
}  // namespace kernel_selector
