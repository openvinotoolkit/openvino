// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multinomial_kernel_ref.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

std::size_t GetBatchNum(const multinomial_params &params) {
    return params.inputs[0].Feature().v == 1 ? 1 : params.inputs[0].Batch().v;
}

std::size_t GetClassSize(const multinomial_params &params) {
    return params.inputs[0].Feature().v == 1 ? params.inputs[0].Batch().v : params.inputs[0].Feature().v;
}

std::size_t GetSamplesSize(const multinomial_params &params) {
    return params.inputs[0].Feature().v == 1 ? params.outputs[0].Batch().v : params.outputs[0].Feature().v;
}

} // anonymous namespace

ParamsKey MultinomialKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData MultinomialKernelRef::SetDefault(const multinomial_params &params,
                                                    const optional_params&) const {
    CommonDispatchData dispatch_data {};
    dispatch_data.gws = {GetBatchNum(params), params.with_replacement ? GetSamplesSize(params) : 1, 1};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo);
    return dispatch_data;
}

JitConstants MultinomialKernelRef::GetJitConstants(const multinomial_params &params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstants(
        {
        MakeJitConstant("BATCH_NUM", GetBatchNum(params)),
        MakeJitConstant("CLASS_SIZE", GetClassSize(params)),
        MakeJitConstant("SAMPLES_SIZE", GetSamplesSize(params)),
        MakeJitConstant("WITH_REPLACEMENT", params.with_replacement)
        });
    return jit;
}

KernelsData MultinomialKernelRef::GetKernelsData(const Params &params,
                                                 const optional_params &options) const {
    if (!Validate(params, options)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<multinomial_params>(params);
    multinomial_params &new_params = dynamic_cast<multinomial_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    auto multinomial_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, multinomial_specific_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo,
                     kernelName, jit, entry_point, EXE_MODE_DEFAULT, false, false, 2);

    return {kernel_data};
}

bool MultinomialKernelRef::Validate(const Params &p, const optional_params &o) const {
    if (p.GetType() != KernelType::MULTINOMIAL || o.GetType() != KernelType::MULTINOMIAL) {
        return false;
    }
    const multinomial_params &params = dynamic_cast<const multinomial_params&>(p);
    if (params.inputs.empty())
        return false;
    return true;
}

} // namespace kernel_selector
