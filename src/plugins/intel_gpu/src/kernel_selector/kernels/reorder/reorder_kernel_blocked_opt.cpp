// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_blocked_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
static constexpr size_t preferred_vec_size = 8;
static constexpr size_t preferred_array_size = 32;
static inline size_t GetGroupSize(const DataTensor& tensor);

ParamsKey ReorderKernelBlockedOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::UINT16);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT16);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::UINT16);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

ReorderKernelBase::DispatchData ReorderKernelBlockedOpt::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;
    dispatchData.gws = {GetGroupSize(params.inputs[0]), 1, 1};
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}


bool ReorderKernelBlockedOpt::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p))
        return false;

    const reorder_params& params = static_cast<const reorder_params&>(p);
    if (GetGroupSize(params.inputs[0]) == 1)
        return false;

    if (!params.fused_ops.empty())
        return false;

    if (params.surface_input || params.inputs[0].GetDType() == Datatype::BF16 )
        return false;

    if (params.mode != MeanSubtractMode::NONE)
        return false;

    auto compare_tensors = [](const DataTensor& input, const DataTensor& output) -> bool {
        // Check all parameters except DataType
        auto& input_dims = input.GetDims();
        auto& output_dims = output.GetDims();
        bool same = input.GetLayout() == output.GetLayout() &&
                    input.GetPaddedVal() == output.GetPaddedVal() &&
                    input.GetViewOffset() == output.GetViewOffset() &&
                    input_dims.size() == output_dims.size();
        for (size_t i = 0; i < input_dims.size(); i++) {
            same &= input_dims[i].v == output_dims[i].v &&
                    input_dims[i].pad.before == output_dims[i].pad.before &&
                    input_dims[i].pad.after == output_dims[i].pad.after &&
                    input_dims[i].pitch == output_dims[i].pitch;
        }

        return same;
    };

    auto& input = params.inputs[0];
    auto& output = params.outputs[0];
    auto& input_dims = input.GetDims();
    auto& output_dims = output.GetDims();
    if (input_dims.size() != output_dims.size() || !compare_tensors(input, output)) {
        return false;
    }

    for (size_t i = 0 ; i < input_dims.size(); i++) {
        if (input_dims[i].pad.is_dynamic || output_dims[i].pad.is_dynamic)
            return false;
    }

    return true;
}

JitConstants ReorderKernelBlockedOpt::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    if (params.truncate) {
        jit.AddConstant(MakeJitConstant("CONVERT_TRUNCATE", true));
    }

    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    jit.AddConstant(MakeJitConstant("VEC_SIZE", preferred_vec_size));
    jit.AddConstant(MakeJitConstant("ARRAY_SIZE", preferred_array_size));
    jit.AddConstant(MakeJitConstant("ELEMENTS_NUM", preferred_vec_size * preferred_array_size));

    return jit;
}

KernelsData ReorderKernelBlockedOpt::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
}

void ReorderKernelBlockedOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const reorder_params&>(params);
        auto dispatchData = ReorderKernelBlockedOpt::SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");

        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsPriority ReorderKernelBlockedOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

static inline size_t GetGroupSize(const DataTensor& tensor) {
    size_t size = tensor.PhysicalSize();
    size_t each_item = (preferred_vec_size * preferred_array_size);
    return (Align(size, each_item) / each_item);
}
}  // namespace kernel_selector
