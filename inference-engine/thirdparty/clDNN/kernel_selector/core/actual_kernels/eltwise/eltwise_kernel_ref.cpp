// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// y ou may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey EltwiseKernelRef::GetSupportedKey() const {
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
    k.EnableEltwiseStride();
    k.EnableEltwiseBroadcast();
    return k;
}

bool EltwiseKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(p, o)) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants EltwiseKernelRef::GetJitConstants(const eltwise_params& params) const {
    auto jit = EltwiseKernelBase::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"d4", "d3", "d2", "d1"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"d5", "d4", "d3", "d2", "d1"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 6) {
            idx_order = {"d6", "d5", "d4", "d3", "d2", "d1"};
        }

        if (!params.layoutBased && !params.int8_quantization && !params.broadcast && CheckInputsOutputNoPitchSameDims(params)) {
            FusedOpsConfiguration conf = {"", {"d1"}, "res", input_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::ENABLED, IndexType::LINEAR_OFFSET};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            FusedOpsConfiguration conf =  {"", idx_order, "res", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    return jit;
}
}  // namespace kernel_selector
