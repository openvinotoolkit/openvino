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


#include "eltwise_kernel_fs_b_yx_fsv32.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

ParamsKey EltwiseKernel_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableBatching();
    return k;
}

JitConstants EltwiseKernel_fs_b_yx_fsv32::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, true);
}

bool EltwiseKernel_fs_b_yx_fsv32::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& output = ewParams.output;
    const auto count = output.PhysicalSize();

    const bool bSupportedCount = (count % 8) == 0;

    bool bCheckSizes = true;
    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        // allow only the same input sizes or scalars, without pitches
        if (!(ewParams.inputs[0] == ewParams.inputs[i] && ewParams.inputs[i] == ewParams.output) && ewParams.inputs[i].PhysicalSize() != 1)
            bCheckSizes = false;
    }

    // TODO: add support to this implementation when user requests input values updates
    bool bCheckUpdateInput = true;
    if (!ewParams.updateInputIds.empty())
        bCheckUpdateInput = false;

    // TODO: add support for reading from output buffer and using its values in computation
    bool bCheckUseOutput = true;
    for (size_t op = 0; op < ewParams.operations.size(); op++) {
        for (size_t input_idx = 0; input_idx < ewParams.operations[op].inputs.size(); input_idx++) {
            if (ewParams.operations[op].inputs[input_idx].mode == EltwiseInputMode::OUTPUT_BUFFER) {
                bCheckUseOutput = false;
                break;
            }
        }
    }

    if (!bCheckSizes || !bSupportedCount || !bCheckUpdateInput || !bCheckUseOutput) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernel_fs_b_yx_fsv32::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    std::string jit;

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);

    try {
        auto cldnn_jit = GetJitConstants(newParams);
        jit = CreateJit(kernelName, cldnn_jit, entry_point);
    } catch (const std::runtime_error&) {
        return KernelsData();
    }

    auto& kernel = kd.kernels[0];

    auto& input = newParams.inputs[0];

    size_t batches = input.Batch().v;
    size_t featuresRoundedUp = Align(input.Feature().v, 32);
    size_t y = input.Y().v;
    size_t x = input.X().v;
    size_t global_size = featuresRoundedUp * batches * x * y;

    kernel.workGroups.global = {std::max(global_size / 8, (size_t)1), 1, 1};
    kernel.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.workGroups.global, params.engineInfo);

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    kd.estimatedTime = FORCE_PRIORITY_8;

    return {kd};
}
}  // namespace kernel_selector