// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_shifted_clamp_experimental_kernel_ref.h"

#include <kernel_selector_utils.h>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const scaled_shifted_clamp_experimental_params& params) {
    CommonDispatchData dd;
    const auto& out = params.outputs[0];
    dd.gws = {1, 1, out.Batch().v * out.Feature().v * out.X().v * out.Y().v * out.Z().v * out.W().v};
    dd.lws = GetOptimalLocalWorkGroupSizes(dd.gws, params.engineInfo);
    return dd;
}

}  // namespace

void ScaledShiftedClampExperimentalKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& p = static_cast<const scaled_shifted_clamp_experimental_params&>(params);
        auto dd = SetDefault(p);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dd.gws;
        kd.kernels[0].params.workGroups.local  = dd.lws;
        kd.kernels[0].skip_execution           = KernelData::SkipKernelExecution(p);
    };
}

KernelsData ScaledShiftedClampExperimentalKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    KernelData   kd     = KernelData::Default<scaled_shifted_clamp_experimental_params>(params);
    const auto&  p      = static_cast<const scaled_shifted_clamp_experimental_params&>(params);
    auto         dd     = SetDefault(p);
    auto         entry  = GetEntryPoint(kernelName, p.layerID, params);
    auto         jit_cs = MakeBaseParamsJitConstants(p);
    jit_cs.AddConstant(MakeJitConstant("SCALE", p.scale));
    jit_cs.AddConstant(MakeJitConstant("BIAS", p.bias));
    jit_cs.AddConstant(MakeJitConstant("LO", p.lo));
    jit_cs.AddConstant(MakeJitConstant("HI", p.hi));
    auto jit = CreateJit(kernelName, jit_cs, entry);

    GetUpdateDispatchDataFunc(kd);

    auto&      kernel     = kd.kernels[0];
    const bool is_dynamic = p.has_dynamic_tensors();
    FillCLKernelData(kernel, dd, params.engineInfo, kernelName, jit, entry,
                     EXE_MODE_DEFAULT, false, false, 1, 0, 1, is_dynamic);
    return {kd};
}

KernelsPriority ScaledShiftedClampExperimentalKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey ScaledShiftedClampExperimentalKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

bool ScaledShiftedClampExperimentalKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SCALED_SHIFTED_CLAMP_EXPERIMENTAL)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    return true;
}

}  // namespace kernel_selector
