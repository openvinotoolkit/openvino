// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

bool PermuteKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::PERMUTE) {
        return false;
    }
    const permute_params& params = static_cast<const permute_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op)) {
            return false;
        }
    }

    auto supported_dyn_layouts = {DataLayout::bfyx, DataLayout::bfzyx, DataLayout::bfwzyx};
    if (params.has_dynamic_tensors() && (!layout_is_one_of(params.inputs, supported_dyn_layouts) || !layout_is_one_of(params.outputs, supported_dyn_layouts)))
        return false;

    return true;
}

JitConstants PermuteKernelBase::GetJitConstants(const permute_params& params, const CommonDispatchData&) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

void PermuteKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const permute_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kernel_data.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kernel_data.kernels[0].params.workGroups.global = dispatchData.gws;
        kernel_data.kernels[0].params.workGroups.local = dispatchData.lws;
        kernel_data.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData PermuteKernelBase::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);

    GetUpdateDispatchDataFunc(kd);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    std::pair<std::string, std::string> jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     newParams.is_shape_agnostic);

    return {kd};
}
}  // namespace kernel_selector
