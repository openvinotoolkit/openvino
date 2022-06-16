// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_detection_output_kernel_ref.h"

#include <algorithm>
#include <string>

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ExperimentalDetectronDetectionOutputKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

KernelsPriority ExperimentalDetectronDetectionOutputKernelRef::GetKernelsPriority(const Params&,
                                                                                  const optional_params&) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ExperimentalDetectronDetectionOutputKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT ||
        o.GetType() != KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT) {
        return false;
    }
    return true;
}

constexpr int kBoxesInputIdx = 0;
constexpr int kDeltasInputIdx = 1;
constexpr int kScoresInputIdx = 2;
constexpr int kImInfoInputIdx = 3;
constexpr int kOutputClassesInputIdx = 4;
constexpr int kOutputScoresInputIdx = 5;

constexpr int kOutputIdx = 0;

using DispatchData = CommonDispatchData;

KernelsData ExperimentalDetectronDetectionOutputKernelRef::GetKernelsData(const Params& params,
                                                                          const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    constexpr size_t kKernelCount = 1;
    KernelData kd = KernelData::Default<experimental_detectron_detection_output_params>(params, kKernelCount);
    const auto& eddo_params = static_cast<const experimental_detectron_detection_output_params&>(params);

    DispatchData dispatchData;
    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, eddo_params.engineInfo);

    const auto entryPoint = GetEntryPoint(kernelName, eddo_params.layerID, eddo_params, options, 0);
    JitConstants cldnnJit = MakeBaseParamsJitConstants(eddo_params);

    cldnnJit.AddConstants({
        MakeJitConstant("OUTPUT_INDICES_TYPE", "INPUT4_TYPE"),
    });

    const auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
    CheckDispatchData(kernelName, dispatchData, eddo_params.engineInfo.maxWorkGroupSize);
    kd.kernels[0].params.workGroups.global = dispatchData.gws;
    kd.kernels[0].params.workGroups.local = dispatchData.lws;
    kd.kernels[0].code.kernelString = GetKernelString(kernelName, jit, entryPoint, eddo_params.engineInfo);

    kd.kernels[0].params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, kOutputIdx});
    kd.kernels[0].params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kOutputClassesInputIdx});
    kd.kernels[0].params.arguments.push_back({ArgumentDescriptor::Types::INPUT, kOutputScoresInputIdx});

    return {kd};
}

}  // namespace kernel_selector
