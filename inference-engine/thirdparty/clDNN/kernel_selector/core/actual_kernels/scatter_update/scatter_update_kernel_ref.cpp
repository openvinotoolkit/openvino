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

#include "scatter_update_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
static size_t GetScatterUpdateChannelIndex(const scatter_update_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    const size_t dict_size = params.inputs[0].GetDims().size();
    switch (params.axis) {
        case ScatterUpdateAxis::X:
            return dict_size - 1;
        case ScatterUpdateAxis::Y:
            return dict_size - 2;
        case ScatterUpdateAxis::Z:
            return dict_size - 3;
        case ScatterUpdateAxis::W:
            return 2;
        case ScatterUpdateAxis::FEATURE:
            return 1;
        case ScatterUpdateAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.output.GetLayout(), name);
}

ParamsKey ScatterUpdateKernelRef::GetSupportedKey() const {
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

static std::string GetUpdatesIndexOrder(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());

    for (unsigned int i = 0; i < params.inputs[2].GetDims().size() - params.output.GetDims().size(); i++)
        default_order.push_back("0");

    size_t indices_non_empty_dims = GetNonEmptyDimsNumber(params.inputs[1]);
    std::string FYX_indices_size = "(INPUT1_FEATURE_NUM * INPUT1_SIZE_Y * INPUT1_SIZE_X)";
    std::string YX_indices_size = "(INPUT1_SIZE_Y * INPUT1_SIZE_X)";
    std::string X_indices_size = "(INPUT1_SIZE_X)";

    // Shift indices of ScatterUpdate updates input related to Indices dims
    for (size_t i = default_order.size() - 1; i > (axis + indices_non_empty_dims - 1); i--)
        default_order[i] = default_order[i - indices_non_empty_dims + 1];

    // Insert Indices indexes in axis dimention in the Update index order
    for (size_t i = axis; i < (axis + indices_non_empty_dims) && i < default_order.size(); i++) {
        switch(i - axis) {
            case 0:
                default_order[i] = "(OUTPUT_INDEX_ON_AXIS /" + FYX_indices_size + ")";
                break;
            case 1:
                default_order[i] = "((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")/" + YX_indices_size + ")";
                break;
            case 2:
                default_order[i] = "(((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")%" + YX_indices_size + ")/" + X_indices_size + ")";
                break;
            case 3:
                default_order[i] = "(((OUTPUT_INDEX_ON_AXIS %" + FYX_indices_size + ")%" + YX_indices_size + ")%" + X_indices_size + ")";
                break;
        }
    }

    return GetOrderString(default_order);
}

CommonDispatchData ScatterUpdateKernelRef::SetDefault(const scatter_update_params& params, const optional_params&, bool is_second) const {
    CommonDispatchData dispatchData;
    const auto& output = params.output;

    const size_t indices_size = params.inputs[1].LogicalSize();

    switch (params.inputs[0].GetLayout()) {
    case DataLayout::bfyx:
        dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        if (is_second) {
            if (params.axis == ScatterUpdateAxis::BATCH)
                dispatchData.gws[2] = indices_size * output.Feature().v;
            else if (params.axis == ScatterUpdateAxis::FEATURE)
                dispatchData.gws[2] = indices_size * output.Batch().v;
            else if (params.axis == ScatterUpdateAxis::Y)
                dispatchData.gws[1] = indices_size;
            else
                dispatchData.gws[0] = indices_size;
        }
        break;

    case DataLayout::bfzyx:
        dispatchData.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        if (is_second) {
            if (params.axis == ScatterUpdateAxis::BATCH)
                dispatchData.gws[2] = indices_size * output.Feature().v;
            else if (params.axis == ScatterUpdateAxis::FEATURE)
                dispatchData.gws[2] = indices_size * output.Batch().v;
            else if (params.axis == ScatterUpdateAxis::Z)
                dispatchData.gws[1] = indices_size;
            else if (params.axis == ScatterUpdateAxis::Y)
                dispatchData.gws[0] = indices_size * output.X().v;
            else
                dispatchData.gws[0] = indices_size * output.Y().v;
        }
        break;

    case DataLayout::bfwzyx:
        dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        if (is_second) {
            if (params.axis == ScatterUpdateAxis::BATCH)
                dispatchData.gws[2] = indices_size * output.Feature().v;
            else if (params.axis == ScatterUpdateAxis::FEATURE)
                dispatchData.gws[2] = indices_size * output.Batch().v;
            else if (params.axis == ScatterUpdateAxis::Z)
                dispatchData.gws[1] = indices_size * output.W().v;
            else if (params.axis == ScatterUpdateAxis::W)
                dispatchData.gws[1] = indices_size * output.Z().v;
            else if (params.axis == ScatterUpdateAxis::Y)
                dispatchData.gws[0] = indices_size * output.X().v;
            else
                dispatchData.gws[0] = indices_size * output.Y().v;
        }
        break;
    default: break;
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

static std::string GetOutputIndexOnAxis(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    return default_order[axis];
}

static std::vector<std::string> GetVectorSecondOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    default_order[axis] = "convert_int(indices[OUTPUT_INDEX_ON_AXIS])";
    return default_order;
}

static std::string GetSecondIterOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    default_order[axis] = "convert_int(indices[OUTPUT_INDEX_ON_AXIS])";
    return GetOrderString(default_order);
}

JitConstants ScatterUpdateKernelRef::GetJitConstants(const scatter_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("UPDATES_INDEX_ORDER", GetUpdatesIndexOrder(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("SECOND_ITER_OUTPUT_INDEX_ORDER", GetSecondIterOutputIndexOrder(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("OUTPUT_INDEX_ON_AXIS", GetOutputIndexOnAxis(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", GetScatterUpdateChannelIndex(params)));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetVectorSecondOutputIndexOrder(params, GetScatterUpdateChannelIndex(params)), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf1, conf2}));
    }

    return jit;
}

bool ScatterUpdateKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType:: SCATTER_UPDATE || o.GetType() != KernelType::SCATTER_UPDATE) {
        return false;
    }

    const scatter_update_params& params = static_cast<const scatter_update_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData ScatterUpdateKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const scatter_update_params& orgParams = static_cast<const scatter_update_params&>(params);
    const size_t indices_size = orgParams.inputs[1].LogicalSize();
    int start_with_iteration = 0;

    // if dim of output along axis is equal to logical size of indices, we miss copying kernel
    if (orgParams.inputs[0].Extract(orgParams.inputs[0].GetLayout(), Tensor::DataChannelName(orgParams.axis), orgParams.inputs[0].GetDims()).v == indices_size) {
        start_with_iteration = 1;
    }

    KernelData kd = KernelData::Default<scatter_update_params>(params, (2 - start_with_iteration));
    scatter_update_params& newParams = *static_cast<scatter_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    for (int i = start_with_iteration; i < 2; i++) {
        auto dispatchData = SetDefault(newParams, options, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);

        if (i == 1){
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
        }
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i - start_with_iteration];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params));
    }

    kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return {kd};
}
}  // namespace kernel_selector
