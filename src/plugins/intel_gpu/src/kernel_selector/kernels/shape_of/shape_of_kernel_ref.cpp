// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const shape_of_params &params) {
    CommonDispatchData dispatchData;

    dispatchData.gws = { 1, 1, params.input_rank };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace

JitConstants ShapeOfKernelRef::GetJitConstants(const shape_of_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("INPUT_DIMS", params.input_dims));

    return jit;
}

bool ShapeOfKernelRef::SkipKernelExecution(const shape_of_params& params) const {
    return false;
}

void ShapeOfKernelRef::SetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const shape_of_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = SkipKernelExecution(prim_params);
    };
}

KernelsData ShapeOfKernelRef::GetKernelsData(const Params &params, const optional_params &options) const {
    KernelsData kernels_data;
    if (!Validate(params, options))
        return kernels_data;
    kernels_data.push_back(KernelData::Default<shape_of_params>(params));
    KernelData &kernel_data = kernels_data.front();
    auto &derived_params = dynamic_cast<shape_of_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(derived_params);
    auto entry_point = GetEntryPoint(kernelName, derived_params.layerID, params, options);
    auto jit_constants = GetJitConstants(derived_params);
    auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto &clKernelData = kernel_data.kernels[0];
    clKernelData.skip_execution = SkipKernelExecution(derived_params);

    SetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(clKernelData, dispatch_data, params.engineInfo, kernelName, jit, entry_point, EXE_MODE_DEFAULT,
                     false, false, 0, 0, 1, derived_params.inputs[0].is_dynamic());
    return kernels_data;
}

KernelsPriority ShapeOfKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey ShapeOfKernelRef::GetSupportedKey() const {
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
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableDynamicShapesSupport();
    return k;
}

bool ShapeOfKernelRef::Validate(const Params &p, const optional_params &o) const {
    if (p.GetType() != KernelType::SHAPE_OF || o.GetType() != KernelType::SHAPE_OF)
        return false;

    return true;
}

}  // namespace kernel_selector
