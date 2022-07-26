// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_blocked_all_axis.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelBlockedAllAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);

    k.EnableSoftmaxDim(SoftmaxDim::ALL);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    return k;
}

SoftmaxKernelBlockedAllAxis::DispatchData SoftmaxKernelBlockedAllAxis::SetDefault(const softmax_params& params,
                                                                                                const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);
    const auto& in = params.inputs[0];
    dispatchData.gws = {in.Batch().v, in.Feature().v, 1};
    dispatchData.lws = dispatchData.gws;

    return dispatchData;
}

KernelsPriority SoftmaxKernelBlockedAllAxis::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool SoftmaxKernelBlockedAllAxis::Validate(const Params& params, const optional_params& o) const {
    if (!Parent::Validate(params, o)) {
        return false;
    }
    const auto& softmax_params = static_cast<const kernel_selector::softmax_params&>(params);
    return softmax_params.dim == SoftmaxDim::ALL;
}

KernelsData SoftmaxKernelBlockedAllAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
JitConstants SoftmaxKernelBlockedAllAxis::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    const auto& in = params.inputs[0];
    const auto ndims = in.GetDims().size();
    const auto class_num = in.Feature().v * in.Batch().v;
    jit.AddConstant(MakeJitConstant("CLASS_NUM", class_num));
    const std::vector<std::string> idx_order = {"b", "f", ndims == 5 ? "z" : "0", "y", "x"};

    const auto acc_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"",
                                      idx_order,
                                      "res",
                                      acc_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
