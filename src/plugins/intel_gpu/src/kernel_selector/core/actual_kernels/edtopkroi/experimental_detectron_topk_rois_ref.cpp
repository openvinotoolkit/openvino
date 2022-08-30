// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "experimental_detectron_topk_rois_ref.h"

#include <kernel_selector_utils.h>
#include <random>


namespace kernel_selector {

namespace {


CommonDispatchData SetDefault(const experimental_detectron_topk_roi_params &params, const optional_params &) {
    CommonDispatchData dispatchData;
    dispatchData.gws = {params.outputs[0].Batch().v, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

}  // namespace

JitConstants ExperimentalDetectronTopKROIRef::GetJitConstants(const experimental_detectron_topk_roi_params &params) const {
    return MakeBaseParamsJitConstants(params);
}


KernelsData ExperimentalDetectronTopKROIRef::GetKernelsData(const Params &params, const optional_params &options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<experimental_detectron_topk_roi_params>(params);
    const experimental_detectron_topk_roi_params &new_params = dynamic_cast<const experimental_detectron_topk_roi_params &>(*kernel_data.params.get());

    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);

    auto experimental_detectron_topk_roi_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, experimental_detectron_topk_roi_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false,
                     false, 2);

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kernel_data));
    return kernelsData;
}

KernelsPriority ExperimentalDetectronTopKROIRef::GetKernelsPriority(const Params & /*params*/,
                                                                    const optional_params & /*options*/) const {
    return FORCE_PRIORITY_1;
}

ParamsKey ExperimentalDetectronTopKROIRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableOutputLayout(Tensor::bfyx);
    k.EnableInputLayout(Tensor::bfyx);
    k.EnableBatching();
    return k;
}

bool ExperimentalDetectronTopKROIRef::Validate(const Params &params, const optional_params &optionalParams) const {
    if (params.GetType() != KernelType::EXPERIMENTAL_DETECTRON_TOPK_ROIS ||
        optionalParams.GetType() != KernelType::EXPERIMENTAL_DETECTRON_TOPK_ROIS) {
        return false;
    }

    const experimental_detectron_topk_roi_params &new_params = dynamic_cast<const experimental_detectron_topk_roi_params &>(params);
    if (new_params.inputs.size() != 2) {
        return false;
    }
    return true;
}

}  // namespace kernel_selector
