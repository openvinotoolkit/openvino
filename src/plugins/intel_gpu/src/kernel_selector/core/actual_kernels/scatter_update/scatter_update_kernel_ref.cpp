// Copyright (C) 2018-2022 Intel Corporation
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

static inline std::string GetAxisName(size_t size, size_t axis) {
    std::vector<std::string> axis_names;;
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

CommonDispatchData ScatterUpdateKernelRef::SetDefault(const scatter_update_params& params, const optional_params&, bool is_second) const {
    CommonDispatchData dispatchData;
    const auto& output = params.outputs[0];
    if (!is_second) {
        switch (output.GetLayout()) {
            case DataLayout::bfyx:
               dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
               break;
            case DataLayout::bfzyx:
               dispatchData.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
               break;
            case DataLayout::bfwzyx:
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
        switch (output.GetLayout()) {
            case DataLayout::bfyx:
                if (params.axis == ScatterUpdateAxis::BATCH)
                    dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * indices_size};
                else if (params.axis == ScatterUpdateAxis::FEATURE)
                    dispatchData.gws = {output.X().v, output.Y().v, indices_size * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::Y)
                     dispatchData.gws = {output.X().v, indices_size, output.Feature().v * output.Batch().v};
                else if (params.axis == ScatterUpdateAxis::X)
                     dispatchData.gws = {indices_size, output.Y().v, output.Feature().v * output.Batch().v};
                break;
            case DataLayout::bfzyx:
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
            case DataLayout::bfwzyx:
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
    output_order[axis] = "convert_int(indices[OUTPUT_INDEX_ON_AXIS])";
    return output_order;
}

static std::string GetSecondIterOutputIndexOrder(const scatter_update_params& params, size_t axis) {
    auto output_order = GetVectorSecondOutputIndexOrder(params, axis);
    return GetOrderString(output_order);
}

JitConstants ScatterUpdateKernelRef::GetJitConstants(const scatter_update_params& params) const {
    size_t axis_value = GetScatterUpdateChannelIndex(params);

    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("UPDATES_INDEX_ORDER", GetUpdatesIndexOrder(params)));
    jit.AddConstant(MakeJitConstant("SECOND_ITER_OUTPUT_INDEX_ORDER", GetSecondIterOutputIndexOrder(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("OUTPUT_INDEX_ON_AXIS", GetOutputIndexOnAxis(params, GetScatterUpdateChannelIndex(params))));
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", axis_value));
    jit.AddConstant(MakeJitConstant("INDICES_SIZE", params.inputs[1].LogicalSize()));

    auto default_order = GetDefaultOrder(params.outputs[0].GetDims().size());
    size_t dims = default_order.size();
    std::string get_update_idx = "(INPUT2_OFFSET)";
    std::string output_size_feature = "OUTPUT_FEATURE_NUM";
    for (size_t i = 0; i < dims; ++i) {
        if (i >= axis_value) {
            std::string def_pitch = "UPDATES_" + GetAxisName(dims, i) + "_PITCH";
            std::string src_pitch = "(OUTPUT_" + GetAxisName(dims, i) + "_PITCH)";
            jit.AddConstant(MakeJitConstant(def_pitch, src_pitch));
        } else if (i == (axis_value - 1)) {
            std::string def_pitch = "UPDATES_" + GetAxisName(dims, i) + "_PITCH";
            std::string src_pitch = "(OUTPUT_" + GetAxisName(dims, i + 1) + "_PITCH * INDICES_SIZE)";
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
        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(params.outputs[0].GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetVectorSecondOutputIndexOrder(params, GetScatterUpdateChannelIndex(params)),
                                        "val", params.inputs[0].GetDType() };
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

    if (params.outputs[0].PitchesDifferFromLogicalDims() || params.inputs[2].PitchesDifferFromLogicalDims()) {
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
    if (orgParams.inputs[0].Extract(orgParams.inputs[0].GetLayout(), Tensor::DataChannelName(orgParams.axis),
                                    orgParams.inputs[0].GetDims()).v == indices_size) {
        start_with_iteration = 1;
    }

    KernelData kd = KernelData::Default<scatter_update_params>(params, (2 - start_with_iteration));
    scatter_update_params& newParams = *static_cast<scatter_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    for (int i = start_with_iteration; i < 2; i++) {
        auto dispatchData = SetDefault(newParams, options, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options, i);

        if (i == 1) {
            cldnn_jit.AddConstant(MakeJitConstant("IS_SECOND_ITER", "true"));
        }
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        clKernelData& kernel = kd.kernels[i - start_with_iteration];

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params));
    }

    return {kd};
}

KernelsPriority ScatterUpdateKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
