// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

bool EltwiseKernel_fs_b_yx_fsv32::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& output = ewParams.outputs[0];
    const auto count = output.PhysicalSize();

    const bool bSupportedCount = (count % 8) == 0;

    bool bCheckSizes = true;
    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        // allow only the same input sizes or scalars, without pitches
        if (!(ewParams.inputs[0] == ewParams.inputs[i] && ewParams.inputs[i] == ewParams.outputs[0]) && ewParams.inputs[i].PhysicalSize() != 1)
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

    if (IsUnsupportedModeForVecCode(ewParams))
        return false;

    if (!bCheckSizes || !bSupportedCount || !bCheckUpdateInput || !bCheckUseOutput) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernel_fs_b_yx_fsv32::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    std::pair<std::string, std::string> jit;

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);

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

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);

    kernel.params.workGroups.global = {std::max(global_size / 8, (size_t)1), 1, 1};
    kernel.params.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.params.workGroups.global, params.engineInfo);

    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    return {kd};
}

KernelsPriority EltwiseKernel_fs_b_yx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
