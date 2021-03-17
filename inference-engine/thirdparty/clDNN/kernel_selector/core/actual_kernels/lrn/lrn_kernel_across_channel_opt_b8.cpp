// Copyright (c) 2016 Intel Corporation
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


#include "lrn_kernel_across_channel_opt_b8.h"

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannel_b8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableSubGroup();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData LRNKernelAcrossChannel_b8::SetDefault(const lrn_params& params) const {
    CommonDispatchData dispatchData = LRNKernelBase::SetDefault(params);

    dispatchData.gws[0] /= 8;
    dispatchData.lws[0] = 8;  // gws[0] is dividable by 64, so after correction it will be dividable by 8.

    return dispatchData;
}

bool LRNKernelAcrossChannel_b8::Validate(const Params& p, const optional_params& o) const {
    if (!LRNKernelBase::Validate(p, o)) {
        return false;
    }

    const lrn_params& params = static_cast<const lrn_params&>(p);
    const auto& out = params.output;

    const bool bSupportedPitch = params.inputs[0].Batch().pitch == 1 && out.Batch().pitch == 1;
    const bool bSupportedBatch = (out.Batch().v % 8) == 0 && ((out.Batch().v * out.Feature().v) % 64) == 0;

    if (!bSupportedPitch || !bSupportedBatch) {
        return false;
    }

    return true;
}

JitConstants LRNKernelAcrossChannel_b8::GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    const auto& input_dt = params.inputs[0].GetDType();

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", 8));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {
            "",
            {"batch_id", "feature_id", "y", "x"},
            "lrn_result",
            input_dt,
            8,
            LoadType::LT_UNALIGNED,
            BoundaryCheck::DISABLED,
            IndexType::TENSOR_COORD,
            Tensor::DataChannelName::BATCH
        };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

KernelsData LRNKernelAcrossChannel_b8::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority LRNKernelAcrossChannel_b8::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
