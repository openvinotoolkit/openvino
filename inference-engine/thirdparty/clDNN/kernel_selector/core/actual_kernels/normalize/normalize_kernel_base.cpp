// Copyright (c) 2016-2020 Intel Corporation
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


#include "normalize_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants NormalizeKernelBase::GetJitConstants(const normalize_params& np) const {
    JitConstants jit = MakeBaseParamsJitConstants(np);

    jit.AddConstants({
        MakeJitConstant("SCALE_TABLE", np.scaleTable),
        MakeJitConstant("EPSILON", np.epsilon),
        MakeJitConstant(toString(np.normMode), ""),
        MakeJitConstant("THRESHOLD", 0.0001f),
    });

    auto activation_dt = GetActivationType(np);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    if (!np.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "b", "f", "y", "x" };
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt);
        jit.Merge(MakeFusedOpsJitConstants(np, { conf }));
    }

    return jit;
}

NormalizeKernelBase::DispatchData NormalizeKernelBase::SetDefault(const normalize_params& params) const {
    const auto& output = params.output;

    DispatchData dispatchData;
    if (params.normMode == NormalizeMode::WITHIN_SPATIAL) {
        dispatchData.gws = {output.X().v, output.Y().v, output.Batch().v};
    } else {
        dispatchData.gws = {output.Batch().v, 1, 1};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData NormalizeKernelBase::GetCommonKernelsData(const Params& params,
                                                      const optional_params& options,
                                                      float estimated_time) const {
    assert(params.GetType() == KernelType::NORMALIZE);
    if (!Validate(params, options))
        return {};

    const normalize_params& orgParams = static_cast<const normalize_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<normalize_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    kernel.arguments.push_back({ArgumentDescriptor::Types::SCALE_TABLE, 0});

    kd.estimatedTime = estimated_time;

    return {kd};
}

bool NormalizeKernelBase::Validate(const Params& params, const optional_params&) const {
    const normalize_params& orgParams = static_cast<const normalize_params&>(params);

    for (auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype NormalizeKernelBase::GetActivationType(const normalize_params& params) const {
    if (params.output.GetDType() == Datatype::F16)
        return Datatype::F16;
    return Datatype::F32;
}
}  // namespace kernel_selector
