// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey DynamicQuantizeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants DynamicQuantizeKernelRef::GetJitConstants(const dynamic_quantize_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.outputs[0]));

    bool rearrange_scales = false;
    const auto& scales_output_order = params.scales_output_order;
    if (!scales_output_order.empty()) {
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            if (i != scales_output_order[i]) {
                rearrange_scales = true;
                break;
            }
        }
    }

    if (rearrange_scales) {
        const std::array<char, 4> default_dim_order = {'b', 'f', 'y', 'x'};

        std::stringstream ss;
        for (size_t i = 0; i < scales_output_order.size(); i++) {
            ss << default_dim_order[scales_output_order[i]];

            if (i + 1 != scales_output_order.size())
                ss << ", ";
        }

        jit.AddConstant(MakeJitConstant("SCALES_OUTPUT_ORDER", ss.str()));
    }

    jit.AddConstant(MakeJitConstant("ASYMMETRIC_QUANTIZATION", params.use_asymmetric_quantization));
    jit.AddConstant(MakeJitConstant("GROUP_SCALES_WITH_ZP", params.combine_scales_and_zp));

    const auto& group_sizes = params.group_sizes;
    for (size_t i = 0; i < group_sizes.size(); i++) {
        jit.AddConstant(MakeJitConstant("GROUP_SIZE_DIM" + std::to_string(i), group_sizes[i]));
    }

    return jit;
}

CommonDispatchData DynamicQuantizeKernelRef::SetDefault(const dynamic_quantize_params& params) const {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    CommonDispatchData dispatchData;

    OPENVINO_ASSERT(params.outputs[0].GetLayout() == DataLayout::bfyx, "It supports only 4d tensor");

    const auto& group_sizes = params.group_sizes;
    auto batch_size = group_sizes[0] == 1 ? params.outputs[0].Batch().v : 1;
    auto feature_size = group_sizes[1] == 1 ? params.outputs[0].Feature().v : 1;
    auto y_size = group_sizes[2] == 1 ? params.outputs[0].Y().v : 1;
    auto x_size = group_sizes[3] == 1 ? params.outputs[0].X().v : 1;

    dispatchData.gws = {batch_size * feature_size, y_size, x_size};
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

void DynamicQuantizeKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const dynamic_quantize_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;
    };
}

KernelsData DynamicQuantizeKernelRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::DYNAMIC_QUANTIZE);

    if (!Validate(params))
        return {};

    const dynamic_quantize_params& prim_params = static_cast<const dynamic_quantize_params&>(params);
    auto dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<dynamic_quantize_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     static_cast<int>(prim_params.outputs.size()),
                     prim_params.is_shape_agnostic);

    return {kd};
}

KernelsPriority DynamicQuantizeKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

bool DynamicQuantizeKernelRef::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params))
        return false;

    const auto& prim_params = static_cast<const dynamic_quantize_params&>(params);
    if (prim_params.group_sizes.size() != 4)
        return false;

    return true;
}
}  // namespace kernel_selector
