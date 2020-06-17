/*
// Copyright (c) 2019-2020 Intel Corporation
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

#include "gather_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
static size_t GetGatherChannelIndex(const gather_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    switch (params.axis) {
        case GatherAxis::X:
            return 3;
        case GatherAxis::Y:
            return 2;
        case GatherAxis::FEATURE:
            return 1;
        case GatherAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.output.GetLayout(), name);
}

ParamsKey GatherKernelRef::GetSupportedKey() const {
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
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static size_t GetNonEmptyDimsNumber(const DataTensor& data_tensor) {
    if (data_tensor.LogicalSize() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        for (auto& i : data_tensor.GetDims()) {
            if (i.v == 1)
                one_size_dims++;
            else
                break;
        }
        return data_tensor.Dimentions() - one_size_dims;
    } else {
        return 1;
    }
}

static inline std::string GetOrderString(std::vector<std::string>& order) {
    std::string order_str = order[0];
    for (size_t i = 1; i < order.size(); i++)
        order_str += ", " + order[i];
    
    return order_str;
}

static std::string GetDictionaryIndexOrder(const gather_params& params, size_t axis) {
    std::vector<std::string> default_order = { "b", "f", "y", "x" };
    const std::string input_axis_index_macro = "INPUT_AXIS_INDEX";
    const std::string zeroVal = "0";

    size_t dictionary_dims_num = GetNonEmptyDimsNumber(params.inputs[0]);
    size_t indices_dims_num = GetNonEmptyDimsNumber(params.output) - dictionary_dims_num + 1;

    // Shift indices of Gather dictionary input related to output dims
    for (size_t i = axis + 1; i < dictionary_dims_num; i++)
        default_order[i] = default_order[i + indices_dims_num - 1];

    for (size_t i = dictionary_dims_num; i < default_order.size(); i++)
        default_order[i] = zeroVal;

    default_order[axis] = input_axis_index_macro;

    return GetOrderString(default_order);
}

static std::string GetIndecesIdxOrder(const gather_params& params, size_t axis) {
    std::vector<std::string> default_order = { "b", "f", "y", "x" };
    const std::string zero_val = "0";

    size_t indices_dims_num = GetNonEmptyDimsNumber(params.inputs[1]);

    // Shift indices of Gather indices input related to output dims
    for (size_t i = 0; i < indices_dims_num; i++)
        default_order[i] = default_order[axis + i];

    for (size_t i = indices_dims_num; i < default_order.size(); i++)
        default_order[i] = zero_val;

    return GetOrderString(default_order);
}

CommonDispatchData GatherKernelRef::SetDefault(const gather_params& params, const optional_params&) const {
    CommonDispatchData runInfo;
    const auto& output = params.output;

    std::vector<size_t> global = {output.Batch().v, output.Feature().v,output.X().v * output.Y().v};
    std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];
    
    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    return runInfo;
}

JitConstants GatherKernelRef::GetJitConstants(const gather_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("DICTIONARY_INDEX_ORDER", GetDictionaryIndexOrder(params, GetGatherChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("INDICES_INDEX_ORDER", GetIndecesIdxOrder(params, GetGatherChannelIndex(params))));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = { "", {"b", "f", "y", "x"}, "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool GatherKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::GATHER || o.GetType() != KernelType::GATHER) {
        return false;
    }

    const gather_params& params = static_cast<const gather_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData GatherKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<gather_params>(params);
    gather_params& newParams = *static_cast<gather_params*>(kd.params.get());

    auto runInfo = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
