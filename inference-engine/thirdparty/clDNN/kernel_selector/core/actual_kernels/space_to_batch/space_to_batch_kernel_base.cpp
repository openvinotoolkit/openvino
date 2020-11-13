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

#include "space_to_batch_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool SpaceToBatchKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::SPACE_TO_BATCH ||
        o.GetType() != KernelType::SPACE_TO_BATCH) {
        return false;
    }

    const space_to_batch_params& params = static_cast<const space_to_batch_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 6)
        return false;

    return true;
}

CommonDispatchData SpaceToBatchKernelBase::SetDefault(const space_to_batch_params& params, const optional_params&) const {
    const auto& out = params.output;

    CommonDispatchData dispatchData;
    if (out.GetLayout() == DataLayout::b_fs_yx_fsv16 && out.Feature().v % 16 == 0) {
        dispatchData.gws = { out.Batch().v, out.Feature().v, out.Y().v * out.X().v };
        dispatchData.lws = {1, 16, 1};
    } else {
        dispatchData.gws = { out.Batch().v, out.Feature().v, out.W().v * out.Z().v * out.Y().v * out.X().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }

    return dispatchData;
}

JitConstants SpaceToBatchKernelBase::GetJitConstants(const space_to_batch_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto makeJitConstForParam = [](JitConstants& jit, const std::string name, const DimTensor<uint32_t>& args, const size_t default_value) {
        jit.AddConstant(MakeJitConstant(name + "_SIZES", args));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", args.b));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", args.f));
        jit.AddConstant(MakeJitConstant(name + "_Y", args.y));
        jit.AddConstant(MakeJitConstant(name + "_X", args.x));

        if (args.w != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", args.w));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else if(args.z != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", default_value));
        }
    };

    makeJitConstForParam(jit, "BLOCK_SHAPE", params.block_shape, 1);
    makeJitConstForParam(jit, "PADS_BEGIN", params.pads_begin, 0);
    makeJitConstForParam(jit, "PADS_END", params.pads_end, 0);

    return jit;
}

KernelsData SpaceToBatchKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimatedTime) const {
    KernelData kd = KernelData::Default<space_to_batch_params>(params);
    space_to_batch_params& newParams = *static_cast<space_to_batch_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = estimatedTime;

    return { kd };
}
}  // namespace kernel_selector
