// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prior_grid_generator_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const experimental_detectron_prior_grid_generator_params &params) {
    CommonDispatchData dispatchData;

    dispatchData.gws = { params.layer_height, params.layer_width, params.inputs[0].Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace

KernelsData ExperimentalDetectronPriorGridGeneratorKernelRef::GetKernelsData(const Params &params) const {
    KernelsData kernels_data;
    if (!Validate(params))
        return kernels_data;
    kernels_data.push_back(KernelData::Default<experimental_detectron_prior_grid_generator_params>(params));
    KernelData &kernel_data = kernels_data.front();
    auto &derived_params = dynamic_cast<experimental_detectron_prior_grid_generator_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(derived_params);
    auto entry_point = GetEntryPoint(kernelName, derived_params.layerID, params);
    auto jit_constants = GetJitConstants(derived_params);
    auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto &clKernelData = kernel_data.kernels[0];
    FillCLKernelData(clKernelData, dispatch_data, params.engineInfo, kernelName, jit, entry_point);
    return kernels_data;
}

KernelsPriority ExperimentalDetectronPriorGridGeneratorKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey ExperimentalDetectronPriorGridGeneratorKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    return k;
}

bool ExperimentalDetectronPriorGridGeneratorKernelRef::Validate(const Params &p) const {
    if (p.GetType() != KernelType::EXPERIMENTAL_DETECTRON_PRIOR_GRID_GENERATOR)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    auto &params = dynamic_cast<const experimental_detectron_prior_grid_generator_params&>(p);
    if (params.inputs.size() != 1)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants ExperimentalDetectronPriorGridGeneratorKernelRef::GetJitConstants(const experimental_detectron_prior_grid_generator_params& params) const {
    auto jit_constants = MakeBaseParamsJitConstants(params);
    jit_constants.AddConstant(MakeJitConstant("LAYER_WIDTH", params.layer_width));
    jit_constants.AddConstant(MakeJitConstant("STEP_X", params.step_x));
    jit_constants.AddConstant(MakeJitConstant("STEP_Y", params.step_y));
    if (params.flatten) {
        jit_constants.AddConstant(MakeJitConstant("FLATTEN", 1));
    }
    return jit_constants;
}
}  // namespace kernel_selector
