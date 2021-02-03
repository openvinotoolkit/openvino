/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static inline std::string GetOrderString(std::vector<std::string>& order) {
    std::string order_str = order[0];
    for (size_t i = 1; i < order.size(); i++)
        order_str += ", " + order[i];

    return order_str;
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

ScatterNDUpdateKernelRef::DispatchData ScatterNDUpdateKernelRef::SetDefault(const scatter_nd_update_params& params, const optional_params&, bool is_second) const {
    DispatchData dispatchData;

    if (!is_second) {
        const auto& scope = params.output;

    switch (params.inputs[0].GetLayout()) {
    case DataLayout::bfyx:
            dispatchData.gws = { scope.X().v, scope.Y().v, scope.Feature().v * scope.Batch().v };
        break;

    case DataLayout::bfzyx:
            dispatchData.gws = { scope.X().v * scope.Y().v, scope.Z().v, scope.Feature().v * scope.Batch().v };
        break;

    case DataLayout::bfwzyx:
            dispatchData.gws = { scope.X().v * scope.Y().v, scope.Z().v * scope.W().v, scope.Feature().v * scope.Batch().v };
        break;
    default:
        assert(0);
        break;
    }
    }
    else {
        auto indices_rank = params.indices_rank;
        const auto& indices = params.inputs[1];
        auto indices_dims = indices.LogicalDims();

        if (indices_dims.size() > 1)
        {
            std::reverse(indices_dims.begin(), indices_dims.end());
        }

        dispatchData.indicesLastDim = indices_dims[indices_rank - 1];
        size_t indices_set_size = 1;
        for (int i = 0; i < (indices_rank - 1); i++)
        {
            indices_set_size *= indices_dims[i];
        }
        dispatchData.gws = {1, 1, indices_set_size};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

static std::string GetOutputIndex(const scatter_nd_update_params& params) {
    std::string output_index_str;

    const auto& output = params.output;
   
    const size_t B_LEN = output.Feature().v * output.Z().v * output.W().v * output.Y().v * output.X().v;
    const size_t F_LEN = output.Z().v * output.W().v * output.Y().v * output.X().v;
    const size_t Z_LEN = output.W().v * output.Y().v * output.X().v;
    const size_t W_LEN = output.Y().v * output.X().v;
    const size_t Y_LEN = output.X().v;

    output_index_str.append("const uint b_remain = dst_idx % ").append(std::to_string(B_LEN)).append(";");
    output_index_str.append("const uint f_remain = b_remain % ").append(std::to_string(F_LEN)).append(";");
    output_index_str.append("const uint z_remain = f_remain % ").append(std::to_string(Z_LEN)).append(";");
    output_index_str.append("const uint w_remain = z_remain % ").append(std::to_string(W_LEN)).append(";");
    output_index_str.append("const uint y_remain = w_remain % ").append(std::to_string(Y_LEN)).append(";");

    output_index_str.append("const uint b = dst_idx / ").append(std::to_string(B_LEN)).append(";");
    output_index_str.append("const uint f = b_remain / ").append(std::to_string(F_LEN)).append(";");
    output_index_str.append("const uint z = f_remain / ").append(std::to_string(Z_LEN)).append(";");
    output_index_str.append("const uint w = z_remain / ").append(std::to_string(W_LEN)).append(";");
    output_index_str.append("const uint y = w_remain / ").append(std::to_string(Y_LEN)).append(";");
    output_index_str.append("const uint x = y_remain;");

    return output_index_str;
}


JitConstants ScatterNDUpdateKernelRef::GetJitConstants(const scatter_nd_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (!params.fused_ops.empty()) {
        jit.AddConstant(MakeJitConstant("OUTPUT_INDEX_SECOND_KERNEL", GetOutputIndex(params)));

        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf1, conf2}));
    }

    return jit;
}

bool ScatterNDUpdateKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType:: SCATTER_ND_UPDATE || o.GetType() != KernelType::SCATTER_ND_UPDATE) {
        return false;
    }

    const scatter_nd_update_params& params = static_cast<const scatter_nd_update_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

static std::string GetInputBlockND(const scatter_nd_update_params& params)
{
    const auto& input = params.inputs[0];
    auto input_dims = input.LogicalDims();
    std::reverse(input_dims.begin(), input_dims.end());
    while (!input_dims.empty() && input_dims.back() == 1)
    {
        input_dims.pop_back();
    }
    const int rank = (int)input_dims.size();
    std::vector<size_t> block_nd(rank + 1);
    block_nd[rank] = 1;
    for (int idx = (rank - 1); idx >= 0; idx--)
    {
        block_nd[idx] = input_dims[idx] * block_nd[idx + 1];
    }

    std::stringstream s;
    for (int i = 0; i < (rank + 1); i++)
    {
        if (i < rank)
        {
            s << block_nd[i] << ",";
        }
        else
        {
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
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);

        if (i == 1){
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
            cldnn_jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", dispatchData.indicesLastDim));
            cldnn_jit.AddConstant(MakeJitConstant("INPUT_BLOCK_ND", GetInputBlockND(newParams)));
        }
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params));
    }

    return {kd};
}
}  // namespace kernel_selector
