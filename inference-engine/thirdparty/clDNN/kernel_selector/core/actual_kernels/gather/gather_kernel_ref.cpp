// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
static size_t GetGatherChannelIndex(const gather_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    size_t inputSize = params.inputs[0].GetDims().size();

    switch (params.axis) {
        case GatherAxis::X:
            return inputSize - 1;
        case GatherAxis::Y:
            return inputSize - 2;
        case GatherAxis::Z:
            return inputSize - 3;
        case GatherAxis::W:
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

static int64_t GetGatherBatchDim(const gather_params& params) {
    if (params.batch_dim < 0)
        return (int64_t)GetNonEmptyDimsNumber(params.inputs[1]) + params.batch_dim;
    else
        return params.batch_dim;
}

static inline std::string GetOrderString(std::vector<std::string>& order) {
    std::string order_str = order[0];
    for (size_t i = 1; i < order.size(); i++)
        order_str += ", " + order[i];

    return order_str;
}

static inline std::vector<std::string> GetOrder(size_t size) {
    std::vector<std::string> idx_order;
    if (size <= 4) {
        idx_order = {"b", "f", "y", "x"};
    } else if (size == 5) {
        idx_order = {"b", "f", "z", "y", "x"};
    } else if (size == 6) {
        idx_order = {"b", "f", "w", "z", "y", "x"};
    }

    return idx_order;
}

static std::string GetDictionaryIndexOrder(const gather_params& params, size_t axis) {
    std::vector<std::string> idx_order = GetOrder(params.output.GetDims().size());

    const std::string input_axis_index_macro = "INPUT_AXIS_INDEX";
    const std::string zeroVal = "0";

    size_t dictionary_dims_num = GetNonEmptyDimsNumber(params.inputs[0]);
    size_t indices_dims_num = GetNonEmptyDimsNumber(params.output) - dictionary_dims_num + 1;

    // Shift indices of Gather dictionary input related to output dims
    for (size_t i = axis + 1; i < dictionary_dims_num; i++)
        idx_order[i] = idx_order[i + indices_dims_num - 1];

    for (size_t i = dictionary_dims_num; i < idx_order.size(); i++)
        idx_order[i] = zeroVal;

    // Fix size to inputs[0] dims size
    for (size_t i = 0; i < params.output.GetDims().size() - params.inputs[0].GetDims().size(); i++)
        idx_order.pop_back();

    idx_order[axis] = input_axis_index_macro;

    return GetOrderString(idx_order);
}

static std::string GetIndecesIdxOrder(const gather_params& params, size_t axis, int64_t batch_dim) {
    std::vector<std::string> idx_order = GetOrder(params.output.GetDims().size());

    const std::string zero_val = "0";

    size_t indices_dims_num = GetNonEmptyDimsNumber(params.inputs[1]);

    // Shift indices of Gather indices input related to output dims
    for (size_t i = batch_dim; i < indices_dims_num; i++)
        idx_order[i] = idx_order[axis + i - batch_dim];

    for (size_t i = indices_dims_num; i < idx_order.size(); i++)
        idx_order[i] = zero_val;

    // Fix size to inputs[1] dims size
    for (size_t i = 0; i < params.output.GetDims().size() - params.inputs[1].GetDims().size(); i++)
        idx_order.pop_back();

    return GetOrderString(idx_order);
}

CommonDispatchData GatherKernelRef::SetDefault(const gather_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;
    const auto& output = params.output;

    if (output.GetLayout() == DataLayout::bfyx) {
        dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
    } else if (output.GetLayout() == DataLayout::bfzyx) {
        dispatchData.gws = {output.X().v, output.Y().v * output.Z().v, output.Feature().v * output.Batch().v};
    } else {
        dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants GatherKernelRef::GetJitConstants(const gather_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("DICTIONARY_INDEX_ORDER", GetDictionaryIndexOrder(params, GetGatherChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("INDICES_INDEX_ORDER", GetIndecesIdxOrder(params, GetGatherChannelIndex(params), GetGatherBatchDim(params))));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = GetOrder(params.inputs[0].GetDims().size());

        FusedOpsConfiguration conf = { "", idx_order, "val", params.inputs[0].GetDType() };
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

    auto dispatchData = SetDefault(newParams, options);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2, GetFusedPrimitiveInputsCount(params));

    return {kd};
}

KernelsPriority GatherKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
