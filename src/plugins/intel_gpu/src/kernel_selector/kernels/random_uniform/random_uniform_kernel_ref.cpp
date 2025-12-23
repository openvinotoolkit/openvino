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

    const auto& output = params.outputs[0];
    if (output.is_dynamic()) {
        DimensionAccessHelperJit dims(output);
        jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", toVectorMulString({dims.x(),
                                                                                              dims.y(),
                                                                                              dims.z(),
                                                                                              dims.w(),
                                                                                              dims.u(),
                                                                                              dims.v(),
                                                                                              dims.f(),
                                                                                              dims.b()})));
    } else {
        jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", params.outputs[0].LogicalSize()));
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

    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto random_uniform_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, random_uniform_specific_jit, entry_point);

    auto dispatch_data = SetDefault(new_params);
    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false,
                     false, static_cast<int>(new_params.inputs.size()), static_cast<uint32_t>(new_params.fused_ops.size()),
                     static_cast<int>(new_params.outputs.size()), new_params.is_shape_agnostic);

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
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

void RandomUniformKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const random_uniform_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

bool RandomUniformKernelRef::Validate(const Params &params) const {
    if (params.GetType() != KernelType::RANDOM_UNIFORM) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // output shape, min value, max value
    constexpr uint32_t number_of_inputs = 3;
    auto &randomUniformParams = dynamic_cast<const random_uniform_params &>(params);
    if (randomUniformParams.inputs.size() != number_of_inputs) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

}  // namespace kernel_selector
