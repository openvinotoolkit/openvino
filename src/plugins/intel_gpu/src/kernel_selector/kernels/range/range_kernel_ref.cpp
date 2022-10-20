// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const range_params &params) {
    CommonDispatchData dispatchData;

    dispatchData.gws = { 1, 1, params.outputs[0].X().v }; // TODO: these could be split better
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace

KernelsData RangeKernelRef::GetKernelsData(const Params &params, const optional_params &options) const {
    KernelsData kernels_data;
    if (!Validate(params, options))
        return kernels_data;
    kernels_data.push_back(KernelData::Default<range_params>(params));
    KernelData &kernel_data = kernels_data.front();
    auto &derived_params = dynamic_cast<range_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(derived_params);
    auto entry_point = GetEntryPoint(kernelName, derived_params.layerID, params, options);
    auto jit_constants = MakeBaseParamsJitConstants(derived_params);
    auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto &clKernelData = kernel_data.kernels[0];
    FillCLKernelData(clKernelData, dispatch_data, params.engineInfo, kernelName, jit, entry_point, DEFAULT,
                     false, false, 3);
    auto &arguments = clKernelData.params.arguments;
    arguments.erase(arguments.begin()+1); // stop is not used by kernel
    return kernels_data;
}

KernelsPriority RangeKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
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
    return k;
}

bool RangeKernelRef::Validate(const Params &p, const optional_params &o) const {
    if (p.GetType() != KernelType::RANGE || o.GetType() != KernelType::RANGE)
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
