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

static std::string GetUpdatesIndexOrder(const scatter_update_params& params) {
    std::vector<std::string> default_order = GetDefaultOrder(params.inputs[2].GetDims().size());

    return GetOrderString(default_order);
}

CommonDispatchData ScatterUpdateKernelRef::SetDefault(const scatter_update_params& params, const optional_params&, bool is_second) const {
    CommonDispatchData dispatchData;
    const auto& output = params.output;
    const auto& updates = params.inputs[2];
    const auto& tensor = is_second ? updates : output;

    switch (tensor.GetLayout()) {
        case DataLayout::bfyx:
           dispatchData.gws = {tensor.X().v, tensor.Y().v, tensor.Feature().v * tensor.Batch().v};
           break;
        case DataLayout::bfzyx:
           dispatchData.gws = {tensor.X().v * tensor.Y().v, tensor.Z().v, tensor.Feature().v * tensor.Batch().v};
           break;
        case DataLayout::bfwzyx:
           dispatchData.gws = {tensor.X().v * tensor.Y().v, tensor.Z().v * tensor.W().v, tensor.Feature().v * tensor.Batch().v};
           break;
        default:
           throw std::runtime_error("Unsupported combination\n");
           break;
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

static std::string GetOutputIndexOnAxis(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.output.GetDims().size());
    return default_order[axis];
}

static std::vector<std::string> GetIndicesOrder(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> indices_order {"0", "0", "0", "0"};
    auto updates_dim = params.inputs[2].GetDims().size();
    auto updates_order = GetDefaultOrder(updates_dim);
    auto input_dim = params.inputs[0].GetDims().size();
    if (updates_dim == input_dim) {
        indices_order[1] = updates_order[axis];
    } else {
        for (int32_t i = 0; i <= ((int32_t)updates_dim - (int32_t)input_dim); ++i) {
            indices_order[i] = updates_order[axis + i];
        }
    }
    return indices_order;
}

static std::vector<std::string> GetVectorSecondOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    auto output_order  = GetDefaultOrder(params.output.GetDims().size());
    auto update_order  = GetDefaultOrder(params.inputs[2].GetDims().size());
    auto indices_order = GetIndicesOrder(params, axis);

    auto indices_size  = GetNonEmptyDimsNumber(params.inputs[1]);
    // When original input for indice size is 1d of size N, the indices tensor is given as 2d tensor [1/*b*/, N/*f*/].
    // However the updates tensor is made with 1d tensor.
    if ((indices_size == 2) && params.inputs[1].Batch().v == 1)
        indices_size = 1;

    for (int32_t i = axis + 1; i < (int32_t) output_order.size(); ++i) {
      output_order[i] = update_order[i + indices_size - 1];
    }

    for (int32_t i = axis - 1; i >= 0; --i) {
      output_order[i] = update_order[i];
    }
    output_order[axis] = "convert_int(indices[(INPUT1_GET_INDEX(" + GetOrderString(indices_order) + "))])";

    return output_order;
}


static std::string GetSecondIterOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    auto output_order = GetVectorSecondOutputIndexOrder(params, axis);
    return GetOrderString(output_order);

}

JitConstants ScatterUpdateKernelRef::GetJitConstants(const scatter_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("UPDATES_INDEX_ORDER", GetUpdatesIndexOrder(params)));
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

    return {kd};
}

KernelsPriority ScatterUpdateKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
