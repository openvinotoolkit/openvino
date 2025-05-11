// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ScatterNDUpdateKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

static inline std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = {"b", "f", "y", "x"};
    } else if (size == 5) {
        default_order = {"b", "f", "z", "y", "x"};
    } else if (size == 6) {
        default_order = {"b", "f", "w", "z", "y", "x"};
    }

    return default_order;
}

ScatterNDUpdateKernelRef::DispatchData
ScatterNDUpdateKernelRef::SetDefault(const scatter_nd_update_params& params, bool is_second) const {
    DispatchData dispatchData;

    if (!is_second) {
        const auto& scope = params.outputs[0];
        dispatchData.indicesLastDim = 1;
        dispatchData.gws = { scope.X().v * scope.Y().v, scope.Z().v * scope.W().v, scope.Feature().v * scope.Batch().v };
    } else {
        auto indices_rank = params.indices_rank;
        const auto& indices = params.inputs[1];
        auto indices_dims = indices.LogicalDims();

        if (indices_dims.size() > 1) {
            std::reverse(indices_dims.begin(), indices_dims.end());
        }

        dispatchData.indicesLastDim = indices_dims[indices_rank - 1];
        size_t indices_set_size = 1;
        for (size_t i = 0; i < (indices_rank - 1); i++) {
            indices_set_size *= indices_dims[i];
        }

        dispatchData.gws = {1, 1, indices_set_size};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ScatterNDUpdateKernelRef::GetJitConstants(const scatter_nd_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(params.outputs[0].GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetDefaultOrder(params.outputs[0].GetDims().size()), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf1, conf2}));
    }

    return jit;
}

bool ScatterNDUpdateKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType:: SCATTER_ND_UPDATE) {
        return false;
    }

    const scatter_nd_update_params& params = static_cast<const scatter_nd_update_params&>(p);

    auto indices_rank = params.indices_rank;
    if (indices_rank < 1) {
        return false;
    }

    if (!params.has_dynamic_inputs()) {
        auto input_dims = params.inputs[0].LogicalDims();
        auto indices_dims = params.inputs[1].LogicalDims();
        std::reverse(indices_dims.begin(), indices_dims.end());

        if (indices_dims[indices_rank - 1] > input_dims.size()) {
            return false;
        }
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

static std::string GetInputBlockND(const scatter_nd_update_params& params, size_t num, size_t shape_info_offset, size_t rank) {
    const auto& input = params.inputs[num];

    auto input_dims = input.LogicalDims();
    std::reverse(input_dims.begin(), input_dims.end());
    auto dims = input.GetDims();
    std::reverse(dims.begin(), dims.end());

    std::vector<size_t> block_nd(rank + 1);
    block_nd[rank] = 1;

    std::vector<std::string> block_nd_s(rank + 1);
    block_nd_s[rank] = "1";
    size_t input_offset = shape_info_offset;

    for (int32_t idx = static_cast<int32_t>(rank) - 1; idx >= 0; --idx) {
        block_nd[idx] = input_dims[idx] * block_nd[idx + 1];

        size_t dim_offset = idx < 2 ? idx : (DataTensor::max_rank() - dims.size()) + idx; // convert to idx in default planar format
        block_nd_s[idx] = "(" + toCodeString(dims[idx], input_offset + dim_offset) + "*" + block_nd_s[idx + 1] + ")";
    }

    std::string result;
    if (input.is_dynamic()) {
        for (auto& block : block_nd_s) {
            result += block + ",";
        }
    } else {
        for (size_t block : block_nd) {
            result += toCodeString(block) + ",";
        }
    }
    return result;
}

void ScatterNDUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_nd_update_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == 2, "[GPU] Invalid kernels size for update dispatch data func");

        for (size_t i = 0; i < 2; ++i) {
            auto dispatchData = SetDefault(prim_params, (i == 1));
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = KernelData::SkipKernelExecution(prim_params);

            // Do not skip copy stage if output buffer is not empty or requires modification
            if (i == 0 && prim_params.outputs[0].LogicalSize() != 0 && prim_params.outputs[0] != prim_params.inputs[0])
                kd.kernels[i].skip_execution = false;
        }
    };
}

KernelsData ScatterNDUpdateKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<scatter_nd_update_params>(params, 2);
    scatter_nd_update_params& newParams = *static_cast<scatter_nd_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    GetUpdateDispatchDataFunc(kd);

    // First iter - copy input data to output data
    // Second iter - update values specified by updates at specific index position specified by indices
    for (int i = 0; i < 2; i++) {
        auto dispatchData = SetDefault(newParams, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, i);
        auto inputs_number = i == 0 ? 1 : 3;

        if (i == 1) {
            size_t input0_rank = newParams.inputs[0].LogicalDims().size();
            size_t input2_rank = newParams.inputs[2].LogicalDims().size();
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
            cldnn_jit.AddConstant(MakeJitConstant(
                "INPUT0_BLOCK_ND",
                GetInputBlockND(newParams, 0, newParams.inputs[0].get_dynamic_shape_offset(), input0_rank)));
            cldnn_jit.AddConstant(MakeJitConstant(
                "INPUT1_BLOCK_ND",
                GetInputBlockND(newParams, 1, newParams.inputs[1].get_dynamic_shape_offset(), newParams.indices_rank - 1)));
            cldnn_jit.AddConstant(MakeJitConstant(
                "INPUT2_BLOCK_ND",
                GetInputBlockND(newParams, 2, newParams.inputs[2].get_dynamic_shape_offset(), input2_rank)));

            cldnn_jit.AddConstant(MakeJitConstant("INDICES_RANK", newParams.indices_rank));

            const auto& ind_input = newParams.inputs[1];
            if (ind_input.is_dynamic()) {
                auto dims = ind_input.GetDims();
                std::reverse(dims.begin(), dims.end());

                size_t last_idx = newParams.indices_rank - 1;
                size_t dim_offset = last_idx < 2 ? last_idx : last_idx + DataTensor::max_rank() - newParams.indices_rank;
                auto indices_last_dim = toCodeString(dims[last_idx], dim_offset + (newParams.inputs[0].is_dynamic() ? DataTensor::max_rank() : 0));
                cldnn_jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", indices_last_dim));
            } else {
                cldnn_jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", dispatchData.indicesLastDim));
            }
        }
        std::pair<std::string, std::string> jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                         "", false, false, inputs_number, GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);
    }

    return {kd};
}
}  // namespace kernel_selector
