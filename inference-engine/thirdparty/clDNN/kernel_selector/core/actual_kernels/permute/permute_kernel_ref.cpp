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


#include "permute_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey PermuteKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants PermuteKernelRef::GetJitConstants(const permute_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    std::vector<std::string> in_idx;
    std::vector<std::string> out_idx;
    switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
        case 6: in_idx = {"b", "f", "x", "y", "z", "w" }; break;
        case 5: in_idx = {"b", "f", "x", "y", "z" }; break;
        default: in_idx = {"b", "f", "x", "y" }; break;
    }

    assert(params.order.size() == in_idx.size());
    for (auto& o : params.order) {
        out_idx.push_back(in_idx[o]);
    }

    std::string input_order = in_idx[0] + "," + in_idx[1];
    std::string output_order = out_idx[0] + "," + out_idx[1];

    for (size_t i = in_idx.size() - 1; i > 1; i--) {
        input_order += "," + in_idx[i];
        output_order += "," + out_idx[i];
    }

    jit.AddConstant(MakeJitConstant("IN_IDX", "INPUT0_GET_INDEX(" + input_order + ")"));
    jit.AddConstant(MakeJitConstant("OUT_IDX", "OUTPUT_GET_INDEX(" + output_order + ")"));

    if (!params.fused_ops.empty()) {
        if (out_idx.size() == 4)
            std::swap(out_idx[2], out_idx[3]);
        else if (out_idx.size() == 5)
            std::swap(out_idx[2], out_idx[4]);
        else if (out_idx.size() == 6) {
            std::swap(out_idx[2], out_idx[5]);
            std::swap(out_idx[3], out_idx[4]);
        }

        FusedOpsConfiguration conf = {"", out_idx, "input_var", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData PermuteKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::PERMUTE);

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    const auto& in = newParams.inputs[0];
    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = {in.X().v, in.Y().v * in.Z().v * in.W().v, in.Feature().v * in.Batch().v};
    kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global, params.engineInfo);
    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc(1, false, false, false, false, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
