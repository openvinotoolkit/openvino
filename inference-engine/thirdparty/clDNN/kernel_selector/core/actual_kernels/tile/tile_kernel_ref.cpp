// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tile_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
static int32_t GetTileChannelIndex(const tile_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;
    switch (params.axis) {
        case TileAxis::X:
            name = Tensor::DataChannelName::X;
            break;
        case TileAxis::Y:
            name = Tensor::DataChannelName::Y;
            break;
        case TileAxis::Z:
            name = Tensor::DataChannelName::Z;
            break;
        case TileAxis::FEATURE:
            name = Tensor::DataChannelName::FEATURE;
            break;
        case TileAxis::BATCH:
            name = Tensor::DataChannelName::BATCH;
            break;
        default:
            break;
    }

    return DataTensor::Channelndex(params.output.GetLayout(), name);
}

ParamsKey TileKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData TileKernelRef::SetDefault(const tile_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    auto in = params.inputs[0];

    size_t inner_size = 1;
    size_t outer_size = 1;

    const int32_t axis = GetTileChannelIndex(params);

    for (int32_t i = 0; i <= axis; i++) {
        inner_size *= in.GetDims()[i].v;
    }

    for (int32_t i = axis + 1; i < static_cast<int32_t>(in.GetDims().size()); i++) {
        outer_size *= in.GetDims()[i].v;
    }

    if (inner_size > 1) {
        dispatchData.gws[0] = outer_size;
        dispatchData.gws[1] = inner_size;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else {
        dispatchData.gws[0] = Align(outer_size, 16);
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;

        dispatchData.lws[0] = 16;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

JitConstants TileKernelRef::GetJitConstants(const tile_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto in = params.inputs[0];
    auto out = params.output;

    size_t inner_size = 1;
    size_t outer_size = 1;
    size_t axis_pitch = 1;

    const int32_t axis = GetTileChannelIndex(params);

    for (int32_t i = 0; i <= axis; i++) {
        inner_size *= in.GetDims()[i].v;
        axis_pitch *= in.GetDims()[i].LogicalDimPadded();
    }
    for (int32_t i = axis + 1; i < static_cast<int32_t>(in.GetDims().size()); i++) {
        outer_size *= in.GetDims()[i].v;
    }

    jit.AddConstant(MakeJitConstant("TILES", params.tiles));
    jit.AddConstant(MakeJitConstant("AXIS_PITCH", axis_pitch));
    jit.AddConstant(MakeJitConstant("OUTER_SIZE", outer_size));
    if (inner_size == 1) {
        jit.AddConstant(MakeJitConstant("OUTPUT_ELEMENTS", out.LogicalSize()));
        jit.AddConstant(MakeJitConstant("DENSE", 1));
    }
    return jit;
}

KernelsData TileKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::TILE);

    KernelData kd = KernelData::Default<tile_params>(params);
    tile_params& newParams = *static_cast<tile_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
