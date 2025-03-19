// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform_kernel_ref.h"

#include <kernel_selector_utils.h>
#include <random>


namespace kernel_selector {

namespace {

int getStep(const random_uniform_params &params) {
    return BytesPerElement(params.outputs[0].GetDType()) > 4 ? 2 : 4;
}

size_t GetGwsSize(const random_uniform_params &params) {
    size_t shapeSize = params.outputs[0].LogicalSize();
    int step = getStep(params);
    return CeilDiv(shapeSize, step);
}

CommonDispatchData SetDefault(const random_uniform_params &params) {
    CommonDispatchData dispatchData;
    dispatchData.gws = {GetGwsSize(params), 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

}  // namespace

JitConstants RandomUniformKernelRef::GetJitConstants(const random_uniform_params &params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    if (params.global_seed == 0 && params.op_seed == 0) {
        jit.AddConstant(MakeJitConstant("GLOBAL_SEED", std::random_device {}()));
    } else {
        jit.AddConstant(MakeJitConstant("GLOBAL_SEED", params.global_seed));
    }

    jit.AddConstant(MakeJitConstant("OP_SEED", params.op_seed));
    jit.AddConstant(MakeJitConstant("OUTPUT_STEP", getStep(params)));
    return jit;
}


KernelsData RandomUniformKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<random_uniform_params>(params);
    const random_uniform_params &new_params = dynamic_cast<const random_uniform_params &>(*kernel_data.params.get());

    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);

    auto random_uniform_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, random_uniform_specific_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false,
                     false, 3);

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kernel_data));
    return kernelsData;
}

KernelsPriority RandomUniformKernelRef::GetKernelsPriority(const Params & /*params*/) const {
    return FORCE_PRIORITY_1;
}

ParamsKey RandomUniformKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    return k;
}

bool RandomUniformKernelRef::Validate(const Params &params) const {
    if (params.GetType() != KernelType::RANDOM_UNIFORM) {
        return false;
    }

    // output shape, min value, max value
    constexpr uint32_t number_of_inputs = 3;
    auto &randomUniformParams = dynamic_cast<const random_uniform_params &>(params);
    if (randomUniformParams.inputs.size() != number_of_inputs) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
