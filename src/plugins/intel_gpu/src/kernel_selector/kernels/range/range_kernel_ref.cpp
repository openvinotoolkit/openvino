// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const range_params &params) {
    CommonDispatchData dispatchData;

    const auto& out = params.outputs[0];
    dispatchData.gws = { 1, 1, out.Batch().v * out.Feature().v * out.X().v * out.Y().v * out.W().v * out.Z().v }; // TODO: these could be split better
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace

void RangeKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const range_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData RangeKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params))
        return {};

    KernelData kernel_data = KernelData::Default<range_params>(params);
    const auto& prim_params = static_cast<const range_params&>(params);

    auto dispatch_data = SetDefault(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit_constants = MakeBaseParamsJitConstants(prim_params);
    auto jit = CreateJit(kernelName, jit_constants, entry_point);

    GetUpdateDispatchDataFunc(kernel_data);

    auto &clKernelData = kernel_data.kernels[0];
    bool is_dynamic = prim_params.has_dynamic_tensors();
    FillCLKernelData(clKernelData, dispatch_data, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 3, 0, 1, is_dynamic);

    auto &arguments = clKernelData.params.arguments;
    arguments.erase(arguments.begin() + 1 + static_cast<int>(is_dynamic)); // stop is not used by kernel

    return {kernel_data};
}

KernelsPriority RangeKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey RangeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT16);
    k.EnableInputDataType(Datatype::UINT16);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT16);
    k.EnableOutputDataType(Datatype::UINT16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

bool RangeKernelRef::Validate(const Params &p) const {
    if (p.GetType() != KernelType::RANGE)
        return false;

    auto &params = dynamic_cast<const range_params&>(p);
    if (params.inputs.size() != 3)
        return false;

    for (auto &input : params.inputs)
        if (input.LogicalSize() != 1)
            return false;
    return true;
}

}  // namespace kernel_selector
