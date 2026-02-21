// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max_kernel_opt.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey SegmentMaxKernelOpt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

SegmentMaxKernelOpt::DispatchData SegmentMaxKernelOpt::SetDefault(const segment_max_params& params) const {
    DispatchData dispatchData;
    const auto& out = params.outputs[0];

    if (out.is_dynamic()) {
        // Fallback for build time — update_dispatch_data_func will patch at runtime.
        dispatchData.gws = {1, 1, 1};
    } else {
        // gws[0] = inner_dim_size  (feature * z * y * x)
        // gws[1] = num_segments    (batch)
        // gws[2] = 1
        size_t inner_dim = out.Feature().v * out.Z().v * out.Y().v * out.X().v;
        size_t num_segments = out.Batch().v;
        dispatchData.gws = {inner_dim, num_segments, 1};
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

void SegmentMaxKernelOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const segment_max_params&>(params);
        const auto& out = prim_params.outputs[0];

        size_t inner_dim = out.Feature().v * out.Z().v * out.Y().v * out.X().v;
        size_t num_segments = out.Batch().v;

        auto& kernel = kd.kernels[0];
        kernel.params.workGroups.global = {inner_dim, num_segments, 1};
        kernel.params.workGroups.local =
            GetOptimalLocalWorkGroupSizes(kernel.params.workGroups.global, prim_params.engineInfo);
        kernel.skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData SegmentMaxKernelOpt::GetKernelsData(const Params& params) const {
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
                     2,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}

KernelsPriority SegmentMaxKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    // Higher priority than the reference kernel (FORCE_PRIORITY_9).
    return FORCE_PRIORITY_7;
}

}  // namespace kernel_selector
