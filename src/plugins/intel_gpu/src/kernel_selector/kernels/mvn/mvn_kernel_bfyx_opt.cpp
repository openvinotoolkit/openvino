// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"
#include "../mvn_rms_scheduling_utils.h"

#include <algorithm>
#include <cctype>
#include <limits>
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
    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        dispatchData.dataSetSize = input.X().v * input.Y().v * input.Z().v;
        dispatchData.dataSetsCount = input.Batch().v * input.Feature().v;
    } else {
        dispatchData.dataSetSize = input.X().v * input.Y().v * input.Z().v * input.Feature().v;
        dispatchData.dataSetsCount = input.Batch().v;
    }

    if (dispatchData.dataSetSize != 0 && dispatchData.dataSetsCount != 0) {
        dispatchData.gws[0] = 1;
        dispatchData.gws[1] = dispatchData.dataSetsCount;
        dispatchData.gws[2] = 1;

        const auto policy = MvnSchedulingPolicy::GetAdaptivePolicy(dispatchData.dataSetSize);
        dispatchData.lws[0] = MvnSchedulingPolicy::GetGeneralizedLws(dispatchData.dataSetSize, max_lws, policy.target_items);
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
        dispatchData.itemsNum = dispatchData.dataSetSize / dispatchData.lws[0];

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
        std::string lws_0 = "get_local_size(0)";
        size_t stack_size = MvnSchedulingPolicy::kMaxRegisterStack;
        bool reread_input = true;
        bool is_static_lws = false;
        size_t static_data_set_size = 0;
        bool has_static_data_set_size = false;
        if (StaticDimExpressionParser::IsDecimalNumber(data_set_size)) {
            static_data_set_size = std::stoul(data_set_size);
            has_static_data_set_size = true;
        } else {
            has_static_data_set_size = StaticDimExpressionParser::TryFoldMulExpression(data_set_size, static_data_set_size);
        }

        if (has_static_data_set_size) {
            const auto policy = MvnSchedulingPolicy::GetAdaptivePolicy(static_data_set_size);
            const size_t lws = MvnSchedulingPolicy::GetGeneralizedLws(static_data_set_size,
                                                                       dispatchData.maxSlmSize,
                                                                       policy.target_items);
            const size_t required_stack = MvnSchedulingPolicy::GetStackSize(static_data_set_size, lws);
            lws_0 = std::to_string(lws);
            stack_size = std::min(required_stack, policy.stack_cap);
            reread_input = required_stack > policy.stack_cap;
            is_static_lws = true;
        }
        jit.AddConstants({
            MakeJitConstant("LWS_IS_STATIC", is_static_lws),
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("DATA_SET_SIZE", data_set_size),
            MakeJitConstant("DATA_SETS_COUNT", data_set_count),
            MakeJitConstant("MVN_STACK_SIZE", stack_size),
            MakeJitConstant("MVN_REREAD_INPUT", reread_input),
        });
    } else {
        const auto policy = MvnSchedulingPolicy::GetAdaptivePolicy(dispatchData.dataSetSize);
        const size_t stack_size = MvnSchedulingPolicy::GetStackSize(dispatchData.dataSetSize, dispatchData.lws[0]);
        jit.AddConstants({
            MakeJitConstant("LWS_IS_STATIC", true),
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("DATA_SETS_COUNT", dispatchData.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE", dispatchData.dataSetSize),
            MakeJitConstant("MVN_STACK_SIZE", std::min(stack_size, policy.stack_cap)),
            MakeJitConstant("MVN_REREAD_INPUT", stack_size > policy.stack_cap),
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

        auto boundary_check = BoundaryCheck::DISABLED;
        if (params.has_dynamic_tensors()) {
            boundary_check = BoundaryCheck::ENABLED;
        } else {
            for (const auto& fused_op : params.fused_ops) {
                if (!fused_op.output_tensor.SameDims(params.outputs[0])) {
                    boundary_check = BoundaryCheck::ENABLED;
                    break;
                }
            }
        }
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt, 1, LoadType::LT_UNALIGNED, boundary_check);
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
