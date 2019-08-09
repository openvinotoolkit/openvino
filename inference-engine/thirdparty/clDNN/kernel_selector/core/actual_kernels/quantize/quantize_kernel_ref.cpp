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


#include <iostream>
#include "quantize_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey QuantizeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData QuantizeKernelRef::SetDefault(const quantize_params& params, const optional_params&) const {
    CommonDispatchData runInfo;

    auto output = params.output;

    runInfo.gws0 = output.Batch().v;
    runInfo.gws1 = params.packed_binary_output ? CeilDiv(output.Feature().v, 32) : output.Feature().v;
    runInfo.gws2 = Align(output.X().v * output.Y().v, 16);

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = 16;

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants QuantizeKernelRef::GetJitConstants(const quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("PACKED_BINARY_OUTPUT", params.packed_binary_output));
    assert(params.inputs.size() == 5);
    if (params.packed_binary_output) {
        if ((params.inputs[3].LogicalSize() == 1 && params.inputs[4].LogicalSize() == 1) ||
            (params.inputs[3].LogicalSize() == params.inputs[3].Batch().v &&
             params.inputs[4].LogicalSize() == params.inputs[4].Batch().v)) {
            jit.AddConstant(MakeJitConstant("SINGLE_OUT_VAL", 1));

        } else if (params.inputs[3].LogicalSize() == params.output.Feature().v &&
                   params.inputs[4].LogicalSize() == params.output.Feature().v) {
            jit.AddConstant(MakeJitConstant("PER_CHANNEL_OUT_VAL", 1));
        } else {
            throw std::runtime_error("Unsupported const blob shape in node " + params.layerID);
        }
    }
    jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_NUM_PACKED", CeilDiv(params.output.Feature().v, 32)));
    jit.AddConstant(MakeJitConstant("OC_BLOCK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("LEVELS", params.levels));

    return jit;
}

KernelsData QuantizeKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::QUANTIZE);

    KernelData kd = KernelData::Default<quantize_params>(params);
    quantize_params& newParams = *static_cast<quantize_params*>(kd.params.get());

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = {runInfo.gws0, runInfo.gws1, runInfo.gws2};
    kernel.workGroups.local = {runInfo.lws0, runInfo.lws1, runInfo.lws2};
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc(static_cast<int>(newParams.inputs.size()), false, false, false, false);

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
