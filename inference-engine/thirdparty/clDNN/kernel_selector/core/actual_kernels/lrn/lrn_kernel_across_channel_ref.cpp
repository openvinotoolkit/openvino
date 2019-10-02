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


#include "lrn_kernel_across_channel_ref.h"

namespace kernel_selector {
ParamsKey LRNKernelAcrossChannelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::ACROSS_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    return k;
}

CommonDispatchData LRNKernelAcrossChannelRef::SetDefault(const lrn_params& params) const {
    CommonDispatchData runInfo = LRNKernelBase::SetDefault(params);

    if (params.inputs[0].GetLayout() == DataLayout::bfyx) {
        const auto& out = params.output;
        runInfo.gws0 = Align(out.X().v, 32);
        runInfo.gws1 = out.Y().v;
        runInfo.gws2 = out.Feature().v * out.Batch().v;

        runInfo.lws0 = 32;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;
    }

    return runInfo;
}

KernelsData LRNKernelAcrossChannelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
}
}  // namespace kernel_selector