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


#include "lrn_kernel_across_channel_multiple_features_fsv16.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannelMultipleFeaturesFSV16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData LRNKernelAcrossChannelMultipleFeaturesFSV16::SetDefault(const lrn_params& params) const {
    CommonDispatchData dispatchData = LRNKernelBase::SetDefault(params);

    const auto& out = params.output;
    const unsigned int alignment = 16;

    dispatchData.gws = { Align(out.Feature().v, alignment),
                         out.X().v,
                         out.Y().v * out.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    dispatchData.efficiency = FORCE_PRIORITY_6;

    return dispatchData;
}

JitConstants LRNKernelAcrossChannelMultipleFeaturesFSV16::GetJitConstants(const lrn_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = LRNKernelBase::GetJitConstants(params, dispatchData);
    const auto& input_dt = params.inputs[0].GetDType();

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id", "y", "x"}, "lrn_result", input_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
