// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "topk_rois_ref.h"

#include <kernel_selector_utils.h>
#include <random>


namespace kernel_selector {

namespace {


CommonDispatchData SetDefault(const experimental_detectron_topk_roi_params &params) {
    CommonDispatchData dispatchData;
    dispatchData.gws = {params.outputs[0].Batch().v, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

}  // namespace

JitConstants ExperimentalDetectronTopKROIRef::GetJitConstants(const experimental_detectron_topk_roi_params &params) const {
    return MakeBaseParamsJitConstants(params);
}


KernelsData ExperimentalDetectronTopKROIRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<experimental_detectron_topk_roi_params>(params);
    const experimental_detectron_topk_roi_params &new_params = dynamic_cast<const experimental_detectron_topk_roi_params &>(*kernel_data.params.get());

    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);

    auto experimental_detectron_topk_roi_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, experimental_detectron_topk_roi_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false,
                     false, 2);

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kernel_data));
    return kernelsData;
}

KernelsPriority ExperimentalDetectronTopKROIRef::GetKernelsPriority(const Params & /*params*/) const {
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
    k.EnableInputLayout(Tensor::bfyx);
    k.EnableInputLayout(Tensor::b_fs_yx_fsv16);
    k.EnableInputLayout(Tensor::b_fs_yx_fsv32);
    k.EnableInputLayout(Tensor::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(Tensor::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(Tensor::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(Tensor::bfyx);
    k.EnableOutputLayout(Tensor::b_fs_yx_fsv16);
    k.EnableOutputLayout(Tensor::b_fs_yx_fsv32);
    k.EnableOutputLayout(Tensor::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(Tensor::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(Tensor::bs_fs_yx_bsv32_fsv16);
    k.EnableBatching();
    k.EnableTensorPitches();
    return k;
}

bool ExperimentalDetectronTopKROIRef::Validate(const Params &params) const {
    if (params.GetType() != KernelType::EXPERIMENTAL_DETECTRON_TOPK_ROIS) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const experimental_detectron_topk_roi_params &new_params = dynamic_cast<const experimental_detectron_topk_roi_params &>(params);
    if (new_params.inputs.size() != 2) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    return true;
}

}  // namespace kernel_selector
