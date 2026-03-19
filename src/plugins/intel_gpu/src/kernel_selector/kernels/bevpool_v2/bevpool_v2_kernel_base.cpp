// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2_kernel_base.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

JitConstants BevPoolV2KernelBase::GetJitConstants(const bevpool_v2_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("INPUTS_COUNT", params.inputs.size()),
        MakeJitConstant("INPUT_CHANNELS", params.input_channels),
        MakeJitConstant("OUTPUT_CHANNELS", params.output_channels),
        MakeJitConstant("IMAGE_WIDTH", params.image_width),
        MakeJitConstant("IMAGE_HEIGHT", params.image_height),
        MakeJitConstant("FEATURE_WIDTH", params.feature_width),
        MakeJitConstant("FEATURE_HEIGHT", params.feature_height),
        MakeJitConstant("X_BOUND_MIN", params.x_bound_min),
        MakeJitConstant("X_BOUND_MAX", params.x_bound_max),
        MakeJitConstant("X_BOUND_STEP", params.x_bound_step),
        MakeJitConstant("Y_BOUND_MIN", params.y_bound_min),
        MakeJitConstant("Y_BOUND_MAX", params.y_bound_max),
        MakeJitConstant("Y_BOUND_STEP", params.y_bound_step),
        MakeJitConstant("Z_BOUND_MIN", params.z_bound_min),
        MakeJitConstant("Z_BOUND_MAX", params.z_bound_max),
        MakeJitConstant("Z_BOUND_STEP", params.z_bound_step),
        MakeJitConstant("D_BOUND_MIN", params.d_bound_min),
        MakeJitConstant("D_BOUND_MAX", params.d_bound_max),
        MakeJitConstant("D_BOUND_STEP", params.d_bound_step),
    });

    return jit;
}

void BevPoolV2KernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const bevpool_v2_params&>(params);
        auto dispatch_data = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatch_data.gws;
        kd.kernels[0].params.workGroups.local = dispatch_data.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

BevPoolV2KernelBase::DispatchData BevPoolV2KernelBase::SetDefault(const bevpool_v2_params& params) {
    DispatchData dispatch_data;
    const uint32_t interval_count = params.inputs.size() > 3 ? static_cast<uint32_t>(params.inputs[3].LogicalSize() / 3) : 1;
    dispatch_data.gws[0] = Align(params.output_channels, static_cast<uint32_t>(16));
    dispatch_data.gws[1] = std::max(interval_count, static_cast<uint32_t>(1));
    dispatch_data.gws[2] = 1;
    dispatch_data.lws[0] = std::min(dispatch_data.gws[0], static_cast<size_t>(16));
    dispatch_data.lws[1] = 1;
    dispatch_data.lws[2] = 1;

    return dispatch_data;
}

KernelsData BevPoolV2KernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BEVPOOL_V2);

    const auto& prim_params = static_cast<const bevpool_v2_params&>(params);
    auto dispatch_data = SetDefault(prim_params);
    auto kernel_data = KernelData::Default<bevpool_v2_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kernel_data);

    auto& kernel = kernel_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     static_cast<uint32_t>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params),
                     static_cast<uint32_t>(prim_params.outputs.size()),
                     prim_params.is_shape_agnostic);

    return {kernel_data};
}

}  // namespace kernel_selector
