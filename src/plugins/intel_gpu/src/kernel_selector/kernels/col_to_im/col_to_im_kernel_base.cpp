// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col_to_im_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool ColToImKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::COL_TO_IM) {
        return false;
    }

    const col_to_im_params& params = static_cast<const col_to_im_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

JitConstants ColToImKernelBase::GetJitConstants(const col_to_im_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("STRIDE", params.stride));
    jit.AddConstant(MakeJitConstant("DILATION", params.dilation));
    jit.AddConstant(MakeJitConstant("PAD_BEGIN", params.padding_begin));
    jit.AddConstant(MakeJitConstant("PAD_END", params.padding_end));

    return jit;
}

KernelsData ColToImKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<col_to_im_params>(params);
    col_to_im_params& newParams = *static_cast<col_to_im_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    return { kd };
}
}  // namespace kernel_selector
