// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

template<typename T>
static void makeJitConstForParam(JitConstants& jit, const std::string name, const T& vec) {
    jit.AddConstant(MakeJitConstant(name + "_SIZES", vec));
    jit.AddConstant(MakeJitConstant(name + "_BATCH", vec[0]));
    jit.AddConstant(MakeJitConstant(name + "_FEATURE", vec[1]));
    if (vec.size() == 6) {  // BFWZYX
        jit.AddConstant(MakeJitConstant(name + "_W", vec[2]));
        jit.AddConstant(MakeJitConstant(name + "_Z", vec[3]));
        jit.AddConstant(MakeJitConstant(name + "_Y", vec[4]));
        jit.AddConstant(MakeJitConstant(name + "_X", vec[5]));
    } else if (vec.size() == 5) {  // BFZYX
        jit.AddConstant(MakeJitConstant(name + "_W", 0));
        jit.AddConstant(MakeJitConstant(name + "_Z", vec[2]));
        jit.AddConstant(MakeJitConstant(name + "_Y", vec[3]));
        jit.AddConstant(MakeJitConstant(name + "_X", vec[4]));
    } else {  // BFYX
        jit.AddConstant(MakeJitConstant(name + "_W", 0));
        jit.AddConstant(MakeJitConstant(name + "_Z", 0));
        jit.AddConstant(MakeJitConstant(name + "_Y", vec[2]));
        jit.AddConstant(MakeJitConstant(name + "_X", vec[3]));
    }
}

static size_t GetUsedOutDimsCount(const strided_slice_params& params) {
    auto dims = params.outputs[0].GetDims();
    size_t first_non_unit_dim = 0; // order is xy(zw)fb, so by default consider that we use all dims
    for (size_t i = 0; i < dims.size(); i++) {
        if (dims[i].v != 1) {
            break;
        }
        first_non_unit_dim = i;
    }
    return dims.size() - first_non_unit_dim;
}

ParamsKey StridedSliceKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

bool StridedSliceKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::STRIDED_SLICE) {
        return false;
    }

    const strided_slice_params& params = static_cast<const strided_slice_params&>(p);
    if (params.inputs.empty())
        return false;

    if (params.outputs[0].Dimentions() > 6 || params.inputs[0].Dimentions() > 6)
        return false;

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    bool shrink_mode = std::find(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), 1) != params.shrink_axis_mask.end();
    if (shrink_mode) {
        size_t shrinked_axes = std::count_if(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), [](const uint8_t& v) {
            return v == 1;
        });
        size_t used_out_dims = GetUsedOutDimsCount(params);

        // Count of actual output dims + count of shrinked axes shouldn't exceed 5 to be able to find input index correctly
        if (used_out_dims + shrinked_axes > 6) {
            return false;
        }
    }
    return true;
}

CommonDispatchData StridedSliceKernelRef::SetDefault(const strided_slice_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                     { Tensor::DataChannelName::FEATURE },
                                                                     { Tensor::DataChannelName::X, Tensor::DataChannelName::Y,
                                                                       Tensor::DataChannelName::Z, Tensor::DataChannelName::W }};

    // If the new_axis_mask is set, then begin, end, and stride are ignored
    // and a new length 1 dimension is adding. Input data just copying to output
    // TODO: remove data copying in case where only shape size changing
    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].W().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

inline std::string GetInputTypeStr(uint32_t idx) {
    return "INPUT" + std::to_string(idx) + "_TYPE";
}

inline std::string GetToInputTypeStr(uint32_t idx) {
    return "TO_" + GetInputTypeStr(idx);
}

inline std::string GetInputIndexStr(uint32_t idx) {
    return "INPUT" + std::to_string(idx) + "_GET_INDEX";
}

JitConstants StridedSliceKernelRef::GetJitConstants(const strided_slice_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    if (params.begin_type == base_params::ArgType::Input || params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("BEGIN_TYPE", GetInputTypeStr(params.GetIndexBegin())));
        jit.AddConstant(MakeJitConstant("TO_BEGIN_TYPE", GetToInputTypeStr(params.GetIndexBegin())));
        jit.AddConstant(MakeJitConstant("BEGIN_GET_INDEX", GetInputIndexStr(params.GetIndexBegin())));
        jit.AddConstant(MakeJitConstant("BEGIN_DIMS", params.begin_dims));
        makeJitConstForParam(jit, "BEGIN", params.begin_mask);
    } else {
        makeJitConstForParam(jit, "SLICE_BEGIN", params.striding_params[0]);
    }
    if (params.end_type == base_params::ArgType::Input || params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("END_TYPE", GetInputTypeStr(params.GetIndexEnd())));
        jit.AddConstant(MakeJitConstant("TO_END_TYPE", GetToInputTypeStr(params.GetIndexEnd())));
        jit.AddConstant(MakeJitConstant("END_GET_INDEX", GetInputIndexStr(params.GetIndexEnd())));
        jit.AddConstant(MakeJitConstant("END_DIMS", params.end_dims));
        makeJitConstForParam(jit, "END", params.end_mask);
    } else {
        makeJitConstForParam(jit, "SLICE_END", params.striding_params[1]);
    }
    if (params.stride_type == base_params::ArgType::Input || params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("STRIDE_TYPE", GetInputTypeStr(params.GetIndexStride())));
        jit.AddConstant(MakeJitConstant("STRIDE_GET_INDEX", GetInputIndexStr(params.GetIndexStride())));
        jit.AddConstant(MakeJitConstant("STRIDE_DIMS", params.stride_dims));
    } else {
        makeJitConstForParam(jit, "SLICE_STEPS", params.striding_params[2]);
    }
    jit.AddConstant(MakeJitConstant(
        "NEW_AXIS_MODE",
        std::find(params.new_axis_mask.begin(), params.new_axis_mask.end(), 1) != params.new_axis_mask.end()));

    std::vector<int> dims_indexes;
    bool ellipsis_mode = std::find(params.ellipsis_mask.begin(), params.ellipsis_mask.end(), 1) != params.ellipsis_mask.end();
    if (ellipsis_mode) {
        size_t ellipsis_pos1 = 0;
        for (size_t i = 0; i < params.ellipsis_mask.size(); i++) {
            if (params.ellipsis_mask[i] == 1) {
                ellipsis_pos1 = i;
                break;
            }
        }

        const size_t output_rank = params.outputs[0].Dimentions();
        const size_t skip_dims_num = output_rank - params.ellipsis_mask.size() + 1;
        int dim_counter = 0;

        for (size_t i = 0; i < ellipsis_pos1; i++)
            dims_indexes.push_back(dim_counter++);

        for (size_t i = 0; i < skip_dims_num; i++)
            dims_indexes.push_back(-1);

        dim_counter++;
        for (size_t i = 0; i < params.ellipsis_mask.size() - ellipsis_pos1 - 1; i++)
            dims_indexes.push_back(dim_counter++);

        OPENVINO_ASSERT(dims_indexes.size() == output_rank, "[GPU] Number of indexes is expected to match with output rank");
    } else {
        dims_indexes.resize(params.outputs[0].Dimentions());
        std::iota(dims_indexes.begin(), dims_indexes.end(), 0);
    }
    makeJitConstForParam(jit, "DIM_IDX", dims_indexes);

    bool shrink_mode = std::find(params.shrink_axis_mask.begin(), params.shrink_axis_mask.end(), 1) != params.shrink_axis_mask.end();
    if (shrink_mode) {
        jit.AddConstant(MakeJitConstant("SHRINK_MODE", true));
        makeJitConstForParam(jit, "SHRINK", params.shrink_axis_mask);
        std::vector<std::string> bfwzyx_in_order;
        if (params.outputs[0].Dimentions() == 6)
            bfwzyx_in_order = {"batch", "feature", "w", "z", "y", "x"};
        else if (params.outputs[0].Dimentions() == 5)
            bfwzyx_in_order = {"batch", "feature", "z", "y", "x"};
        else
            bfwzyx_in_order = {"batch", "feature", "y", "x"};

        // Insert zeroes to indices order for shinked axes
        for (size_t i = 0; i < params.shrink_axis_mask.size(); i++) {
            if (params.shrink_axis_mask[i] == 1) {
                bfwzyx_in_order.insert(bfwzyx_in_order.begin() + i, "0");
            }
        }

        auto get_input_idx_order = [&](std::vector<std::string> bfwzyx_in_order) -> std::string {
            std::string order = bfwzyx_in_order[0] + "," +
                                bfwzyx_in_order[1] + "," +
                                bfwzyx_in_order[2] + "," +
                                bfwzyx_in_order[3] + "," +
                                bfwzyx_in_order[4];
            if (bfwzyx_in_order.size() == 6) order += "," + bfwzyx_in_order[5];
            return order;
        };
        // Erase indices that exceeds 6d tensor. It should be safe, because we check in Validate method that
        // shrinked axes don't result in too big dims count
        while (bfwzyx_in_order.size() > 6) {
            bfwzyx_in_order.pop_back();
        }

        jit.AddConstant(MakeJitConstant("INPUT_INDICES_ORDER", get_input_idx_order(bfwzyx_in_order)));
    }

    return jit;
}

void StridedSliceKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const strided_slice_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData StridedSliceKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<strided_slice_params>(params);
    strided_slice_params& newParams = *static_cast<strided_slice_params*>(kd.params.get());

    assert(params.GetType() == KernelType::STRIDED_SLICE);

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto input = newParams.inputs[0];
    auto input_dt = input.GetDType();

    if (!newParams.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (input.Dimentions() == 6) {
            idx_order = {"b", "f", "w", "z", "y", "x"};
        } else if (input.Dimentions() == 5) {
            idx_order = {"b", "f", "z", "y", "x"};
        } else if (input.Dimentions() == 4) {
            idx_order = {"b", "f", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "input_data", input_dt, 1};
        cldnn_jit.Merge(MakeFusedOpsJitConstants(newParams, {conf}));
    }
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    GetUpdateDispatchDataFunc(kd);

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(newParams.inputs.size()),
                     0, 1, newParams.is_shape_agnostic);

    return {kd};
}

KernelsPriority StridedSliceKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
