// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max_kernel_base.h"

#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {

JitConstants SegmentMaxKernelBase::GetJitConstants(const segment_max_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // FILL_MODE: 0 = ZERO, 1 = LOWEST
    jit.AddConstants({MakeJitConstant("FILL_MODE", params.fill_mode)});

    // Compute the empty segment value at compile time.
    // For ZERO mode use 0, for LOWEST mode use the type's minimum representable value.
    if (params.fill_mode == 0) {
        jit.AddConstants({MakeJitConstant("EMPTY_SEGMENT_VALUE", "0")});
    } else {
        // Use the type-correct minimum value for each output data type.
        jit.AddConstants({MakeJitConstant("EMPTY_SEGMENT_VALUE", "OUTPUT_VAL_MIN")});
    }

    return jit;
}

void SegmentMaxKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const segment_max_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

SegmentMaxKernelBase::DispatchData SegmentMaxKernelBase::SetDefault(const segment_max_params& params) {
    DispatchData dispatchData;
    // Output may be dynamic at build time; use 1 as fallback.
    // update_dispatch_data_func will set actual gws at runtime.
    const auto& out = params.outputs[0];
    dispatchData.gws[0] = out.is_dynamic() ? 1 : out.LogicalSize();
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsData SegmentMaxKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::SEGMENT_MAX);

    const auto& prim_params = static_cast<const segment_max_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<segment_max_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(k_data);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2,                                            // number of inputs (data + segment_ids)
                     GetFusedPrimitiveInputsCount(params),
                     1,                                            // number of outputs
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
