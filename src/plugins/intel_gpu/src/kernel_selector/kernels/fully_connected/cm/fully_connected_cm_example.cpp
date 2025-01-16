// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_cm_example.h"

namespace kernel_selector {
KernelsData FullyConnected_cm_example::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }
    auto options = std::string(" -Qxcm_jit_option=-DPASTokenReduction ");

    KernelData kd = KernelData::Default<fully_connected_params>(params, 1);
    auto& kernel = kd.kernels[0];

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
    kernel.params.workGroups.local = {1, 2, 4};
    kernel.params.workGroups.global = {1, 4, 8};

    std::string kernel_name = "fully_connected_cm_example";
    auto jit = std::pair<std::string, std::string>("\n#define KERNEL_NAME " + kernel_name, "#undef KERNEL_NAME");
    kernel.code.kernelString = GetKernelString("example", jit, kernel_name);
    kernel.code.kernelString->options += options;
    kernel.code.kernelString->batch_compilation = true;
    return {kd};
}
KernelsPriority FullyConnected_cm_example::GetKernelsPriority(const Params& params) const {
    return TUTORIAL_PRIORITY;
}
ParamsKey FullyConnected_cm_example::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableWeightsCompression();
    return k;
}
bool FullyConnected_cm_example::Validate(const Params& p) const {
    return true;
}
}  // namespace kernel_selector
