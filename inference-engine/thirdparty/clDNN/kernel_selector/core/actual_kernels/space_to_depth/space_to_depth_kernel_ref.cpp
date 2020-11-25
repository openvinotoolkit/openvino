/*
// Copyright (c) 2020 Intel Corporation
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

#include "space_to_depth_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey SpaceToDepthKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool SpaceToDepthKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SPACE_TO_DEPTH ||
        o.GetType() != KernelType::SPACE_TO_DEPTH) {
        return false;
    }

    const space_to_depth_params& params = static_cast<const space_to_depth_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

CommonDispatchData SpaceToDepthKernelRef::SetDefault(const space_to_depth_params& params,
                                                     const optional_params&) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = { params.output.Batch().v,
                         params.output.Feature().v,
                         params.output.Z().v * params.output.Y().v * params.output.X().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants SpaceToDepthKernelRef::GetJitConstants(const space_to_depth_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", params.block_size));
    if (params.depth_mode == SpaceToDepthMode::BLOCKS_FIRST)
        jit.AddConstant(MakeJitConstant("BLOCKS_FIRST_MODE", true));
    else
        jit.AddConstant(MakeJitConstant("DEPTH_FIRST_MODE", true));

    auto input = params.inputs[0];
    auto input_dt = input.GetDType();
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (input.Dimentions() == 5) {
            idx_order = {"batch", "feature", "z", "y", "x"};
        } else if (input.Dimentions() == 4) {
            idx_order = {"batch", "feature", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "in_val", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData SpaceToDepthKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<space_to_depth_params>(params);
    space_to_depth_params& newParams = *static_cast<space_to_depth_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
