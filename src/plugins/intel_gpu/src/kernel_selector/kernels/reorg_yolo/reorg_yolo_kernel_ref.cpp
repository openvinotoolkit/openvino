// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorg_yolo_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
namespace {
ReorgYoloKernelRef::DispatchData SetDefault(const reorg_yolo_params& params) {
    ReorgYoloKernelRef::DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& input = params.inputs[0];
    dispatchData.gws = {input.X().v, input.Y().v, input.Feature().v};
    dims_by_gws = {{Tensor::DataChannelName::X},
                   {Tensor::DataChannelName::Y},
                   {Tensor::DataChannelName::FEATURE}};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}
}  // namespace

ParamsKey ReorgYoloKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ReorgYoloKernelRef::GetJitConstants(const reorg_yolo_params& ry) const {
    JitConstants jit = MakeBaseParamsJitConstants(ry);

    jit.AddConstants({
        MakeJitConstant("STRIDE", ry.stride),
    });

    return jit;
}

KernelsData ReorgYoloKernelRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORG_YOLO);
    if (!Validate(params)) {
        return {};
    }

    const reorg_yolo_params& orgParams = static_cast<const reorg_yolo_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<reorg_yolo_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

KernelsPriority ReorgYoloKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

bool ReorgYoloKernelRef::Validate(const Params& p) const {
    const reorg_yolo_params& params = static_cast<const reorg_yolo_params&>(p);
    const auto& input = params.inputs[0];

    if (input.GetDims().size() != 4) {
        return false;
    }

    if (!(input.Feature().v >= params.stride * params.stride
            && input.X().v % params.stride == 0
            && input.Y().v % params.stride == 0)) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
