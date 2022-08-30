// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

ParamsKey TileKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData TileKernelRef::SetDefault(const tile_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::Z, Tensor::DataChannelName::W },
                                                                     { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    auto out = params.outputs[0];

    dispatchData.gws = { out.X().v * out.Y().v, out.Z().v * out.W().v, out.Batch().v * out.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants TileKernelRef::GetJitConstants(const tile_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

KernelsData TileKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::TILE);

    KernelData kd = KernelData::Default<tile_params>(params);
    tile_params& newParams = *static_cast<tile_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

KernelsPriority TileKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
