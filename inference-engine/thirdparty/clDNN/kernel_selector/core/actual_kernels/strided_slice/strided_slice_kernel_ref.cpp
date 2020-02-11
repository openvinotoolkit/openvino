// Copyright (c) 2019 Intel Corporation
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

#include "strided_slice_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey StridedSliceKernelRef::GetSupportedKey() const {
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

CommonDispatchData StridedSliceKernelRef::SetDefault(const strided_slice_params& params, const optional_params&) const {
    CommonDispatchData runInfo;

    // If the new_axis_mask is set, then begin, end, and stride are ignored
    // and a new length 1 dimension is adding. Input data just copying to output
    // TODO: remove data copying in case where only shape size changing
    std::vector<size_t> gws = {params.output.Batch().v, params.output.Feature().v,
                               params.output.Z().v * params.output.Y().v * params.output.X().v};

    auto lws = GetOptimalLocalWorkGroupSizes(gws, params.engineInfo);

    runInfo.gws0 = gws[0];
    runInfo.gws1 = gws[1];
    runInfo.gws2 = gws[2];

    runInfo.lws0 = lws[0];
    runInfo.lws1 = lws[1];
    runInfo.lws2 = lws[2];

    return runInfo;
}

JitConstants StridedSliceKernelRef::GetJitConstants(const strided_slice_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto makeJitConstForParam = [](JitConstants& jit, const std::string name, const std::vector<int32_t> vec) {
        jit.AddConstant(MakeJitConstant(name + "_SIZES", vec));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", vec[0]));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", vec[1]));
        if (vec.size() == 5) {  // BFZYX
            jit.AddConstant(MakeJitConstant(name + "_Z", vec[2]));
            jit.AddConstant(MakeJitConstant(name + "_Y", vec[3]));
            jit.AddConstant(MakeJitConstant(name + "_X", vec[4]));
        } else {  // BFYX
            jit.AddConstant(MakeJitConstant(name + "_Z", 0));
            jit.AddConstant(MakeJitConstant(name + "_Y", vec[2]));
            jit.AddConstant(MakeJitConstant(name + "_X", vec[3]));
        }
    };

    makeJitConstForParam(jit, "SLICE_BEGIN", params.striding_params[0]);
    makeJitConstForParam(jit, "SLICE_END", params.striding_params[1]);
    makeJitConstForParam(jit, "SLICE_STEPS", params.striding_params[2]);

    jit.AddConstant(MakeJitConstant(
        "NEW_AXIS_MODE",
        std::find(params.new_axis_mask.begin(), params.new_axis_mask.end(), 1) != params.new_axis_mask.end()));

    return jit;
}

KernelsData StridedSliceKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<strided_slice_params>(params);
    strided_slice_params& newParams = *static_cast<strided_slice_params*>(kd.params.get());

    assert(params.GetType() == KernelType::STRIDED_SLICE);

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
