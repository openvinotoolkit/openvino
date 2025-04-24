// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
            return (size_t)(dict_size - 1);
        case ScatterUpdateAxis::Y:
            return (size_t)(dict_size - 2);
        case ScatterUpdateAxis::Z:
            return (size_t)(dict_size - 3);
        case ScatterUpdateAxis::W:
            return 2;
        case ScatterUpdateAxis::FEATURE:
            return 1;
        case ScatterUpdateAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.outputs[0].GetLayout(), name);
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
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
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

static inline std::string GetAxisName(size_t size, size_t axis) {
    std::vector<std::string> axis_names;
    if (size <= 4) {
        axis_names = {"BATCH", "FEATURE", "Y", "X"};
    } else if (size == 5) {
        axis_names = {"BATCH", "FEATURE", "Z", "Y", "X"};
    } else if (size == 6) {
        axis_names = {"BATCH", "FEATURE", "W", "Z", "Y", "X"};
    }
    return axis_names[axis];
}

static std::string GetUpdatesIndexOrder(const scatter_update_params& params) {
    std::vector<std::string> default_order = GetDefaultOrder(params.outputs[0].GetDims().size());
    return GetOrderString(default_order);
}

CommonDispatchData ScatterUpdateKernelRef::SetDefault(const scatter_update_params& params, bool is_second) const {
    CommonDispatchData dispatchData;
    const auto& output = params.outputs[0];
    const auto rank = output.GetDims().size();

    if (!is_second) {
        switch (rank) {
            case 4:
               dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
               break;
            case 5:
               dispatchData.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
               break;
            case 6:
               dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
               break;
            default:
               throw std::runtime_error("Unsupported combination\n");
               break;
        }
    } else {
        // second iteration
        // Each work item is for each tensor in input2.
        // Not using input2's shape info directly, because the input2's shape might be reordered from the reordering pass.
        // Instead, we reconsider update2's dimension with input1's shape which is shrinked as 1d.
        // e.g., axis = b, input0(10, 9, 10, 9, 10) && input1(4, 2) => input2(8, 9, 10, 9, 10
        const size_t indices_size = params.inputs[1].LogicalSize();
        switch (rank) {
            case 4:
                if (params.axis == ScatterUpdateAxis::BATCH)
                    dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * indices_size};
                else if (params.axis == ScatterUpdateAxis::FEATURE)
                    dispatchData.gws = {output.X().v, output.Y().v, indices_size * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Y)
                     dispatchData.gws = {output.X().v, indices_size, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::X)
                     dispatchData.gws = {indices_size, output.Y().v, output.Feature().v * output.Batch().v};
                break;
            case 5:
                if (params.axis == ScatterUpdateAxis::BATCH)
                    dispatchData.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * indices_size};
                else if (params.axis == ScatterUpdateAxis::FEATURE)
                    dispatchData.gws = {output.X().v * output.Y().v, output.Z().v, indices_size * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Z)
                    dispatchData.gws = {output.X().v * output.Y().v, indices_size, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Y)
                    dispatchData.gws = {output.X().v * indices_size, output.Z().v, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::X)
                    dispatchData.gws = {indices_size * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
                break;
            case 6:
                if (params.axis == ScatterUpdateAxis::BATCH)
                    dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * indices_size};
                else if (params.axis == ScatterUpdateAxis::FEATURE)
                    dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, indices_size * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::W)
                    dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * indices_size, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Z)
                    dispatchData.gws = {output.X().v * output.Y().v, indices_size * output.W().v, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Y)
                    dispatchData.gws = {output.X().v * indices_size, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::X)
                    dispatchData.gws = {indices_size * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
               break;
            default:
               throw std::runtime_error("Unsupported combination\n");
               break;
        }
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

static std::string GetOutputIndexOnAxis(const scatter_update_params& params, size_t axis) {
    std::vector<std::string> default_order = GetDefaultOrder(params.outputs[0].GetDims().size());
    return default_order[axis];
}

static std::vector<std::string> GetVectorSecondOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    auto output_order = GetDefaultOrder(params.outputs[0].GetDims().size());
    output_order[axis] = "index_by_axis";
    return output_order;
}

static std::string GetSecondIterOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    auto output_order = GetVectorSecondOutputIndexOrder(params, axis);
    return GetOrderString(output_order);
}

static std::vector<std::string> GetPlanarPitches(const Tensor::NDims& dims) {
    const auto ndims = dims.size();
    std::vector<std::string> pitches(ndims);
    size_t pitch = 1;
    for (size_t i = 0; i < ndims; ++i) {
        pitches[i] = std::to_string(pitch);
        pitch *= dims[i].v;
    }
    std::reverse(pitches.begin(), pitches.end());
    return pitches;
}

static std::vector<std::string> GetDynamicPitches(const Tensor::NDims& dims, size_t tensor_idx) {
    std::vector<std::string> pitches(dims.size());
    size_t shape_info_rank = DataTensor::max_rank();
    std::string pitch = "1";
    for (size_t i = 0; i < pitches.size(); ++i) {
        size_t bf_idx = dims.size() - i - 1;
        size_t dim_offset = bf_idx > 1 ? shape_info_rank - i - 1 : bf_idx;

        pitches[i] = pitch;
        pitch += "*" + toCodeString(dims[i], tensor_idx * shape_info_rank + dim_offset);
    }
    std::reverse(pitches.begin(), pitches.end());
    return pitches;
}

JitConstants ScatterUpdateKernelRef::GetJitConstants(const scatter_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    size_t axis_value = GetScatterUpdateChannelIndex(params);

    const auto blocked_layout = !(SimpleLayout(params.inputs[0].GetLayout()) &&
                                  SimpleLayout(params.inputs[1].GetLayout()) &&
                                  SimpleLayout(params.inputs[2].GetLayout()));

    if (blocked_layout) {
        jit.AddConstant(MakeJitConstant("BLOCKED_LAYOUT", "1"));
    }

    jit.AddConstant(MakeJitConstant("UPDATES_INDEX_ORDER", GetUpdatesIndexOrder(params)));
    jit.AddConstant(MakeJitConstant("SECOND_ITER_OUTPUT_INDEX_ORDER",
                                    GetSecondIterOutputIndexOrder(params, static_cast<size_t>(GetScatterUpdateChannelIndex(params)))));
    jit.AddConstant(MakeJitConstant("OUTPUT_INDEX_ON_AXIS", GetOutputIndexOnAxis(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", axis_value));

    const auto& input1 = params.inputs[1];
    if (input1.is_dynamic()) {
        const auto& input1_dims = input1.GetDims();
        size_t input_offset = DataTensor::max_rank();
        std::string indices_size = "(";

        for (size_t i = 0; i < input1_dims.size(); ++i) {
            size_t bf_idx = input1_dims.size() - i - 1;
            size_t dim_offset = bf_idx > 1 ? input_offset - i - 1 : bf_idx;

            indices_size += toCodeString(input1_dims[i], input_offset + dim_offset) + "*";
        }
        indices_size.back() = ')';
        jit.AddConstant(MakeJitConstant("INDICES_SIZE", indices_size));
    } else {
        jit.AddConstant(MakeJitConstant("INDICES_SIZE", params.inputs[1].LogicalSize()));
    }

    std::vector<std::string> pitches;
    const auto& output = params.outputs[0];
    if (output.is_dynamic()) {
        size_t tensor_idx = params.inputs.size() + GetFusedPrimitiveInputsCount(params);
        for (auto input : params.inputs) {
            if (!input.is_dynamic())
                tensor_idx--;
        }
        for (auto fused_op : params.fused_ops) {
            if (!fused_op.output_tensor.is_dynamic())
                tensor_idx--;
        }
        pitches = GetDynamicPitches(output.GetDims(), tensor_idx);
    } else {
        pitches = GetPlanarPitches(output.GetDims());
    }
    auto default_order = GetDefaultOrder(output.GetDims().size());

    size_t dims = default_order.size();
    std::string get_update_idx = "(INPUT2_OFFSET)";
    for (size_t i = 0; i < dims; ++i) {
        if (i >= axis_value) {
            std::string def_pitch = "UPDATES_" + GetAxisName(dims, i) + "_PITCH";
            std::string src_pitch = "(" + pitches[i] + ")";
            jit.AddConstant(MakeJitConstant(def_pitch, src_pitch));
        } else if (i == (axis_value - 1)) {
            std::string def_pitch = "UPDATES_" + GetAxisName(dims, i) + "_PITCH";
            std::string src_pitch = "(" + pitches[i + 1] + " * INDICES_SIZE)";
            jit.AddConstant(MakeJitConstant(def_pitch, src_pitch));
        } else { // i < axis_value - 1
            std::string def_pitch = "UPDATES_" + GetAxisName(dims, i) + "_PITCH" + "";
            std::string output_size_name;
            if (i == 0)
                output_size_name = "OUTPUT_FEATURE_NUM";
            else
                output_size_name = "OUTPUT_SIZE_" + GetAxisName(dims, i + 1);
            std::string src_pitch = "(UPDATES_" + GetAxisName(dims, i + 1) + "_PITCH * " + output_size_name + ")";
            jit.AddConstant(MakeJitConstant(def_pitch, src_pitch));
        }
        get_update_idx = get_update_idx + " + (" + default_order[i] + ")*(UPDATES_" + GetAxisName(dims, i) + "_PITCH)";
    }
    jit.AddConstant(MakeJitConstant("GET_UPDATES_INDEX(idx_order)", get_update_idx));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(output.GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetVectorSecondOutputIndexOrder(params, GetScatterUpdateChannelIndex(params)),
                                        "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf1, conf2}));
    }

    return jit;
}

bool ScatterUpdateKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType:: SCATTER_UPDATE) {
        return false;
    }

    const scatter_update_params& params = static_cast<const scatter_update_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

void ScatterUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_update_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == 2, "[GPU] Invalid kernels size for update dispatch data func");

        for (size_t i = 0; i < 2; ++i) {
            auto dispatchData = SetDefault(prim_params, i == 1);
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = KernelData::SkipKernelExecution(prim_params);
        }
    };
}

KernelsData ScatterUpdateKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const scatter_update_params& orgParams = static_cast<const scatter_update_params&>(params);
    const auto& input0 = orgParams.inputs[0];
    const auto& input1 = orgParams.inputs[1];

    int start_with_iteration = 0;
    if (!orgParams.has_dynamic_inputs()) {
        // if dim of output along axis is equal to logical size of indices, we miss copying kernel
        const size_t indices_size = input1.LogicalSize();
        if (input0.Extract(input0.GetLayout(), Tensor::DataChannelName(orgParams.axis), input0.GetDims()).v == indices_size) {
            start_with_iteration = 1;
        }
    }

    KernelData kd = KernelData::Default<scatter_update_params>(params, (2 - start_with_iteration));
    scatter_update_params& newParams = *static_cast<scatter_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    GetUpdateDispatchDataFunc(kd);

    for (size_t i = start_with_iteration; i < 2; ++i) {
        auto dispatchData = SetDefault(newParams, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, i);

        if (i == 1) {
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
        }
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i - start_with_iteration];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                         "", false, false, 3, GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);
    }

    return {kd};
}

KernelsPriority ScatterUpdateKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
