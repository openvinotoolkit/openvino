// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey GroupNormalizationKernelBfyxOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

GroupNormalizationKernelBfyxOpt::Parent::DispatchData GroupNormalizationKernelBfyxOpt::SetDefault(
    const group_normalization_params &params) const {
    DispatchData dispatchData;

    const auto& input = params.inputs[0];

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    dispatchData.maxSlmSize = max_lws;
    if (!params.has_dynamic_tensors()) {
        dispatchData.dataSetSize = input.Feature().v / params.num_groups * input.X().v * input.Y().v * input.Z().v;
        dispatchData.dataSetsCount = input.Batch().v * params.num_groups;

        // start with 1 thread per data set
        dispatchData.gws[0] = 1;
        dispatchData.gws[1] = dispatchData.dataSetsCount;
        dispatchData.gws[2] = 1;
        dispatchData.itemsNum = dispatchData.dataSetSize;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
        // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
        // reads.
        // WA: itemsNum value has been adjusted less than or equal to 8 to increase the number of work items.
        while ((dispatchData.itemsNum > 8 || dispatchData.lws[0] < dispatchData.itemsNum) && (2 * dispatchData.lws[0] <= max_lws)) {
            dispatchData.lws[0] *= 2;
            dispatchData.itemsNum /= 2;
        }

        dispatchData.gws[0] = dispatchData.lws[0];
        dispatchData.leftovers = dispatchData.dataSetSize % dispatchData.lws[0];
    }
    return dispatchData;
}

JitConstants GroupNormalizationKernelBfyxOpt::GetJitConstants(const group_normalization_params &params,
                                                              GroupNormalizationKernelBase::DispatchData dispatchData) const {
    auto jit = GroupNormalizationKernelBase::GetJitConstants(params);

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelperJit dims(input);
        std::string data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f() + "/" + std::to_string(params.num_groups)});
        std::string data_set_count = toVectorMulString({dims.b(), std::to_string(params.num_groups)});
        const std::string lws_0 = "get_local_size(0)";
        jit.AddConstants({
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("SLM_SIZE", dispatchData.maxSlmSize),
            MakeJitConstant("DATA_SET_SIZE", data_set_size),
            MakeJitConstant("DATA_SETS_COUNT", data_set_count),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("ITEMS_NUM", dispatchData.itemsNum),
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("SLM_SIZE", dispatchData.lws[0]),
            MakeJitConstant("DATA_SETS_COUNT", dispatchData.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE", dispatchData.dataSetSize),
            MakeJitConstant("LEFTOVERS", dispatchData.leftovers),
        });
    }
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                            "(data_set_idx % OUTPUT_FEATURE_NUM)",
                            "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X)",
                            "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
        } else if (params.inputs[0].GetDims().size() == 5) {
            idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                            "(data_set_idx % OUTPUT_FEATURE_NUM)",
                            "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                            "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                            "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
        }
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

void GroupNormalizationKernelBfyxOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const group_normalization_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData GroupNormalizationKernelBfyxOpt::GetKernelsData(const Params &params) const {
    assert(params.GetType() == KernelType::GROUP_NORMALIZATION);

    if (!Validate(params))
        return {};

    const group_normalization_params& orgParams = static_cast<const group_normalization_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<group_normalization_params>(params);

    auto finalKernelName = GetKernelName(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params);
    auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     3,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    return {kd};
}

KernelsPriority GroupNormalizationKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
} // namespace kernel_selector
