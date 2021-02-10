/*
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
*/

#include "reorg_yolo_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

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
ReorgYoloKernelRef::DispatchData SetDefault(const reorg_yolo_params& params) {
    ReorgYoloKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[0];
    if (input.GetLayout() == DataLayout::bfyx) {
        dispatchData.gws = {input.X().v, input.Y().v, input.Feature().v};
    } else {
        dispatchData.gws = {input.Feature().v * input.Batch().v, input.X().v, input.Y().v};
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}
KernelsData ReorgYoloKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REORG_YOLO);
    const reorg_yolo_params& orgParams = static_cast<const reorg_yolo_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<reorg_yolo_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}

KernelsPriority ReorgYoloKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
