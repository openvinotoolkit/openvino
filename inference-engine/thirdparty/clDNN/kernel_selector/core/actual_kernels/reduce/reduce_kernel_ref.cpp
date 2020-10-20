/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "reduce_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {
ParamsKey ReduceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData ReduceKernelRef::SetDefault(const reduce_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = { params.output.X().v * params.output.Y().v,
                         params.output.Z().v * params.output.W().v,
                         params.output.Batch().v * params.output.Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ReduceKernelRef::GetJitConstants(const reduce_params& params) const {
    auto jit = ReduceKernelBase::GetJitConstants(params);

    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetFinalAccumulatorType(params), "FINAL_ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order;
        switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
            case 6: idx_order = {"b", "f", "w", "z", "y", "x" }; break;
            case 5: idx_order = {"b", "f", "z", "y", "x" }; break;
            default: idx_order = {"b", "f", "y", "x" }; break;
        }

        FusedOpsConfiguration conf = {"",
                                      idx_order,
                                      "reduce_result",
                                      input_dt,
                                      1,
                                      LoadType::LT_UNALIGNED,
                                      BoundaryCheck::DISABLED,
                                      IndexType::TENSOR_COORD,
                                      Tensor::DataChannelName::X};

        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData ReduceKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
}

}  // namespace kernel_selector
