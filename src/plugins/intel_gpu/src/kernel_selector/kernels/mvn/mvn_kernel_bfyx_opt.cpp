// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

#include <algorithm>
#include <vector>
#include <string>

namespace kernel_selector {
ParamsKey MVNKernelBfyxOpt::GetSupportedKey() const {
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
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNNormalizeVariance();
    k.EnableDynamicShapesSupport();
    return k;
}

MVNKernelBfyxOpt::Parent::DispatchData MVNKernelBfyxOpt::SetDefault(const mvn_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.inputs[0];

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    dispatchData.maxSlmSize = max_lws;
    if (!params.has_dynamic_tensors()) {
        if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
            dispatchData.dataSetSize = input.X().v * input.Y().v * input.Z().v;
            dispatchData.dataSetsCount = input.Batch().v * input.Feature().v;
        } else {
            dispatchData.dataSetSize = input.X().v * input.Y().v * input.Z().v * input.Feature().v;
            dispatchData.dataSetsCount = input.Batch().v;
        }

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

JitConstants MVNKernelBfyxOpt::GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData dispatchData) const {
    auto jit = MVNKernelBase::GetJitConstants(params, dispatchData);

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelperJit dims(input);
        std::string data_set_size;
        std::string data_set_count;
        if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
            data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z()});
            data_set_count = toVectorMulString({dims.f(), dims.b()});
        } else {
            data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
            data_set_count = dims.b();
        }
        const std::string lws_0 = "get_local_size(0)";
        jit.AddConstants({
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("DATA_SET_SIZE", data_set_size),
            MakeJitConstant("DATA_SETS_COUNT", data_set_count),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("DATA_SETS_COUNT", dispatchData.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE", dispatchData.dataSetSize),
        });
    }
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
                idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                              "(data_set_idx % OUTPUT_FEATURE_NUM)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            } else if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
                idx_order = { "data_set_idx",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            }
        } else if (params.inputs[0].GetDims().size() == 5) {
            if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
                idx_order = { "(data_set_idx / OUTPUT_FEATURE_NUM)",
                              "(data_set_idx % OUTPUT_FEATURE_NUM)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            } else if (params.mvnMode == MVNMode::ACROSS_CHANNELS) {
                idx_order = { "data_set_idx",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z))",
                              "((in_data_set_idx + iteration_in_data_set_offset) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z)",
                              "((in_data_set_idx + iteration_in_data_set_offset) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y)",
                              "((in_data_set_idx + iteration_in_data_set_offset) % OUTPUT_SIZE_X)" };
            }
        }
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData MVNKernelBfyxOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority MVNKernelBfyxOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_7;
}
}  // namespace kernel_selector
