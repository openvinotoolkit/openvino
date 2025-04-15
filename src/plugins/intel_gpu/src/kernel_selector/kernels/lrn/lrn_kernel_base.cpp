// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_base.h"
#include <algorithm>

namespace kernel_selector {
bool LRNKernelBase::Validate(const Params& p) const {
    if (!KernelBaseOpenCL::Validate(p) || p.GetType() != KernelType::LRN) {
        return false;
    }

    const lrn_params& params = static_cast<const lrn_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

JitConstants LRNKernelBase::GetJitConstants(const lrn_params& params, const LRNKernelBase::DispatchData& /*dispatchData*/) const {
    JitConstants mem_consts = MakeBaseParamsJitConstants(params);

    const auto padding = (params.localSize - 1) / 2;

    mem_consts.AddConstants({
        MakeJitConstant("LOCAL_SIZE", params.localSize),
        MakeJitConstant("PADDING", padding),
        MakeJitConstant("ALPHA", params.alpha),
        MakeJitConstant("BETA", params.beta),
        MakeJitConstant("K", params.k),
        MakeJitConstant(toString(params.divMode) + "_KERNEL_DIVIDER", ""),
        MakeJitConstant(toString(params.normMode), ""),
    });

    // auto pad = (np.localSize) / 2;
    auto alpha = params.alpha;
    auto alpha_div_by_size = alpha / params.localSize;
    auto alpha_sign = std::signbit(alpha) ? -1.0f : 1.0f;
    // When used FP16 the value cannot be scaled afterwards by alpha (it must be scaled before computing sum of
    // squares).
    auto alpha_abs_sqrt = std::sqrt(std::abs(alpha));
    auto alpha_div_by_size_abs_sqrt = std::sqrt(std::abs(alpha_div_by_size));

    mem_consts.AddConstants({
        MakeJitConstant("ALPHA_AFTER_FACTORED", params.inputs[0].GetDType() == Datatype::F16 ? alpha_sign : alpha),
        MakeJitConstant("ALPHA_DIV_BY_SIZE", params.inputs[0].GetDType() == Datatype::F16 ? alpha_sign : alpha_div_by_size),
        MakeJitConstant("ALPHA_VAL_FACTOR", params.inputs[0].GetDType() == Datatype::F16 ? alpha_abs_sqrt : 1.0f),
        MakeJitConstant("ALPHA_VAL_FACTOR_DIV_BY_SIZE", params.inputs[0].GetDType() == Datatype::F16 ? alpha_div_by_size_abs_sqrt : 1.0f),
    });

    return mem_consts;
}

LRNKernelBase::DispatchData LRNKernelBase::SetDefault(const lrn_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;

    // Determine global work sizes.
    dispatchData.gws[0] = output.Batch().v * output.Feature().v;  // B, F
    dispatchData.gws[1] = output.X().v;                           // X
    dispatchData.gws[2] = output.Y().v;                           // Y
                             // Find largest positive local work size that is divider for global work size.
    dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
        --dispatchData.lws[0];
    }
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData LRNKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const lrn_params& orgParams = static_cast<const lrn_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<lrn_params>(params);

    auto cldnnJit = GetJitConstants(orgParams, dispatchData);
    auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);
    auto fused_deps_total = GetFusedPrimitiveInputsCount(params);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entryPoint,
                     "",
                     false,
                     false,
                     1,
                     fused_deps_total);

    return {kd};
}
}  // namespace kernel_selector
