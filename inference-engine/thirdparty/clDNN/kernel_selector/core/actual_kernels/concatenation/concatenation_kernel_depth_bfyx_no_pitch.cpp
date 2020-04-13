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


#include "concatenation_kernel_depth_bfyx_no_pitch.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {

ParamsKey ConcatenationKernel_depth_bfyx_no_pitch::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableTensorOffset();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatKernelPerInput();
    return k;
}

bool ConcatenationKernel_depth_bfyx_no_pitch::Validate(const Params& p, const optional_params& o) const {
    if (!ConcatenationKernelBase::Validate(p, o)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    // all inputs have to have same layout
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        if (lt.GetLayout() != same_layout) {
            return false;
        }
    }

    // kernel uses intel_sub_group_block_read that has 4-byte alignment requirement
    if (params.output.GetDType() == Datatype::F16) {
        size_t output_offset = 0;

        for (size_t i = 0; i < params.inputs.size(); i++) {
            for (size_t b = 0; b < params.output.Batch().v; b++) {
                if ((output_offset + b * params.inputs[i].Batch().pitch) % 2 != 0)
                    return false;
            }
            output_offset += params.inputs[i].Batch().pitch;
        }
    }

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernel_depth_bfyx_no_pitch::SetDefault(
    const concatenation_params& params) const {
    DispatchData runInfo = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];
    const auto batch = input.Batch().v;
    runInfo.gws0 = batch;
    runInfo.gws1 = Align(std::max((size_t)1, input.LogicalSize() / batch), 16 * 8) / 8;
    runInfo.gws2 = 1;

    runInfo.lws0 = 1;
    runInfo.lws1 = 16;
    runInfo.lws2 = 1;

    runInfo.efficiency = FORCE_PRIORITY_9;

    return runInfo;
}

KernelsData ConcatenationKernel_depth_bfyx_no_pitch::GetKernelsData(const Params& params,
                                                                    const optional_params& optParams) const {
    return GetCommonKernelsData(params, optParams);
}
}  // namespace kernel_selector
