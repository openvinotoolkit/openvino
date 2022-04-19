// Copyright (C) 2018-2022 Intel Corporation
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

    return DataTensor::Channelndex(params.outputs[0].GetLayout(), name);
}

ParamsKey GatherKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
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

static inline std::string GetGatherMaxIndexDim(const gather_params& params) {
    switch (params.axis) {
    case GatherAxis::BATCH:
        return std::to_string(params.inputs[0].Batch().v);
    case GatherAxis::FEATURE:
        return std::to_string(params.inputs[0].Feature().v);
    case GatherAxis::W:
        return std::to_string(params.inputs[0].W().v);
    case GatherAxis::Z:
        return std::to_string(params.inputs[0].Z().v);
    case GatherAxis::Y:
        return std::to_string(params.inputs[0].Y().v);
    case GatherAxis::X:
        return std::to_string(params.inputs[0].X().v);
    }
    throw "Error";
}

static inline std::string GetOrderString(const std::vector<std::string>& order) {
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
    std::vector<std::string> idx_order = GetOrder(params.outputs[0].GetDims().size());

    size_t dictionary_dims_num = GetNonEmptyDimsNumber(params.inputs[0]);
    size_t indices_dims_num = GetNonEmptyDimsNumber(params.outputs[0]) - dictionary_dims_num + 1;

    // Shift indices of Gather dictionary input related to output dims
    for (size_t i = axis + 1; i < dictionary_dims_num; i++)
        idx_order[i] = idx_order[i + indices_dims_num - 1];

    for (size_t i = dictionary_dims_num; i < idx_order.size(); i++)
        idx_order[i] = "0";

    // Fix size to inputs[0] dims size
    for (size_t i = 0; i < params.outputs[0].GetDims().size() - params.inputs[0].GetDims().size(); i++)
        idx_order.pop_back();

    idx_order[axis] = "INPUT_AXIS_INDEX";

    return GetOrderString(idx_order);
}

static std::string GetIndecesIdxOrder(const gather_params& params, size_t axis, int64_t batch_dim) {
    std::vector<std::string> idx_order = GetOrder(params.outputs[0].GetDims().size());

    // if (params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
    //     swap(idx_order[0], idx_order[1]);
    // }

    size_t indices_dims_num = GetNonEmptyDimsNumber(params.inputs[1]);

    // Shift indices of Gather indices input related to output dims
    for (size_t i = batch_dim; i < indices_dims_num; i++)
        idx_order[i] = idx_order[axis + i - batch_dim];

    for (size_t i = indices_dims_num; i < idx_order.size(); i++)
        idx_order[i] = "0";

    // Fix size to inputs[1] dims size
    for (size_t i = 0; i < params.outputs[0].GetDims().size() - params.inputs[1].GetDims().size(); i++)
        idx_order.pop_back();

    return GetOrderString(idx_order);
}

CommonDispatchData GatherKernelRef::SetDefault(const gather_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;
    const auto& output = params.outputs[0];
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    switch ( out_layout ) {
    case DataLayout::yxfb:
    case DataLayout::fyxb:
    case DataLayout::fs_b_yx_fsv32:
    case DataLayout::bfyx:
    case DataLayout::b_fs_yx_fsv4:
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::bs_fs_yx_bsv4_fsv2:
    case DataLayout::bs_fs_yx_bsv4_fsv4:
    case DataLayout::bs_fs_yx_bsv8_fsv2:
    case DataLayout::bs_fs_yx_bsv8_fsv4:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_yx_bsv32_fsv32:
        dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                   {Tensor::DataChannelName::Y},
                   {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfzyx:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::b_fs_zyx_fsv32:
        dispatchData.gws = {output.X().v, output.Y().v * output.Z().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                    {Tensor::DataChannelName::Y, Tensor::DataChannelName::Z},
                    {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfwzyx:
        dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                    {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                    {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    default:
        throw std::invalid_argument("Unsupported data layout for gather elements primitive");
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants GatherKernelRef::GetJitConstants(const gather_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("DICTIONARY_INDEX_ORDER", GetDictionaryIndexOrder(params, GetGatherChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("INDICES_INDEX_ORDER", GetIndecesIdxOrder(params, GetGatherChannelIndex(params), GetGatherBatchDim(params))));
    if (params.support_neg_ind)
        jit.AddConstant(MakeJitConstant("INDEX_DIM", GetGatherMaxIndexDim(params)));

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
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
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
