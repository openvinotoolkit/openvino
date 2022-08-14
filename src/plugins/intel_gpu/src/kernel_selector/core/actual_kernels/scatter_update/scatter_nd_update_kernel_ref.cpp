// Copyright (C) 2018-2022 Intel Corporation
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
ScatterNDUpdateKernelRef::SetDefault(const scatter_nd_update_params& params, const optional_params&, bool is_second) const {
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

bool ScatterNDUpdateKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType:: SCATTER_ND_UPDATE || o.GetType() != KernelType::SCATTER_ND_UPDATE) {
        return false;
    }

    const scatter_nd_update_params& params = static_cast<const scatter_nd_update_params&>(p);
    auto input_dims = params.inputs[0].LogicalDims();
    auto indices_dims = params.inputs[1].LogicalDims();
    std::reverse(indices_dims.begin(), indices_dims.end());

    auto indices_rank = params.indices_rank;
    if (indices_rank < 1) {
        return false;
    }

    if (indices_dims[indices_rank - 1] > input_dims.size()) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

static std::string GetInputBlockND(const scatter_nd_update_params& params, int num, const int rank) {
    const auto& input = params.inputs[num];
    auto input_dims = input.LogicalDims();
    std::reverse(input_dims.begin(), input_dims.end());
    std::vector<size_t> block_nd(rank + 1);
    block_nd[rank] = 1;
    for (int idx = (rank - 1); idx >= 0; idx--) {
        block_nd[idx] = input_dims[idx] * block_nd[idx + 1];
    }

    std::stringstream s;
    for (int i = 0; i < (rank + 1); i++) {
        if (i < rank) {
            s << block_nd[i] << ",";
        } else {
            s << block_nd[i];
        }
    }
    auto str_result = s.str();
    return str_result;
}

KernelsData ScatterNDUpdateKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<scatter_nd_update_params>(params, 2);
    scatter_nd_update_params& newParams = *static_cast<scatter_nd_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    // First iter - copy input data to output data
    // Second iter - update values specified by updates at specific index position specified by indices
    for (int i = 0; i < 2; i++) {
        auto dispatchData = SetDefault(newParams, options, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options, i);

        if (i == 1) {
            int input0_rank = static_cast<int>(newParams.inputs[0].LogicalDims().size());
            int input2_rank = static_cast<int>(newParams.inputs[2].LogicalDims().size());
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
            cldnn_jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", dispatchData.indicesLastDim));
            cldnn_jit.AddConstant(MakeJitConstant("INPUT0_BLOCK_ND", GetInputBlockND(newParams, 0, input0_rank)));
            cldnn_jit.AddConstant(MakeJitConstant("INPUT1_BLOCK_ND", GetInputBlockND(newParams, 1, newParams.indices_rank - 1)));
            cldnn_jit.AddConstant(MakeJitConstant("INPUT2_BLOCK_ND", GetInputBlockND(newParams, 2, input2_rank)));
            cldnn_jit.AddConstant(MakeJitConstant("INDICES_RANK", newParams.indices_rank));
        }
        std::pair<std::string, std::string> jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params));
    }

    return {kd};
}
}  // namespace kernel_selector
