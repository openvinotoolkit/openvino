// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_blocked_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
static inline size_t SelectVecSizeFromSize(const DataTensor&);
static inline size_t SelectGroupSize(size_t ele_size);

ParamsKey ReorderKernelBlockedOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BF16);
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
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

ReorderKernelBase::DispatchData ReorderKernelBlockedOpt::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;
    size_t global_w_item = std::max(params.inputs[0].PhysicalSize() / SelectVecSizeFromSize(params.inputs[0]), (size_t)1);
    size_t g_size = SelectGroupSize(global_w_item);

    //  dispatchData.gws = {global_w_item, 1, 1};
    std::cout << " >> ReorderKernelBlockedOpt::SetDefault >> params.inputs[0].PhysicalSize() : " << params.inputs[0].PhysicalSize() << std::endl;
    dispatchData.gws = {global_w_item/g_size, 1, 1};
    // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}


bool ReorderKernelBlockedOpt::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p))
        return false;

    const reorder_params& params = static_cast<const reorder_params&>(p);
    std::cout << ">>>>>> " << p.layerID << std::endl;
    if (params.truncate) {
        std::cout << "  -- enabled truncated " << std::endl;
        // return false;
    } else {
        // return false;
    }

    if (SelectVecSizeFromSize(params.inputs[0]) == 1) {
        std::cout << "  -- bad for vector : " << (params.inputs[0].is_dynamic() ? "dynamic" : "no dyn") << std::endl;
        return false;
    } else {
        std::cout << "  -- physical outputs for vector : " << params.inputs[0].PhysicalSize() << std::endl;
    }

    if (!params.fused_ops.empty())
        return false;

    if (params.surface_input || params.inputs[0].GetDType() == Datatype::BF16 )
        return false;

    if (params.mode != MeanSubtractMode::NONE) {
        std::cout << "  -- MeanSubtractMode is not NONE " << std::endl;
        return false;
    }

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

    // std::cout << "  -- info : " << (int)params.mode << " " << (int)params.mean_op << " " << params.meanValues.size() << " "
    //             << " " << params.mean.Dimentions() << " : "
    //             << params.winograd_input_offset_x << " "  << params.winograd_input_offset_y
    //             << " " << params.winograd_nr_tiles_x << " " << params.winograd << " " << params.has_padded_output
    //             << " " << params.surface_input << " " << params.truncate << std::endl;
    // std::cout << "  -- Done : " << p.layerID << std::endl;
    return true;
}

JitConstants ReorderKernelBlockedOpt::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    if (params.truncate) {
        jit.AddConstant(MakeJitConstant("CONVERT_TRUNCATE", true));
    }

    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    size_t vec_size = SelectVecSizeFromSize(params.inputs[0]);
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    size_t global_w_item = std::max(params.inputs[0].PhysicalSize() / vec_size, (size_t)1);
    size_t g_size = SelectGroupSize(global_w_item);
    jit.AddConstant(MakeJitConstant("ITEM_SIZE", g_size));

    std::cout << "  -- " << params.layerID << " >> ITEM_SIZE : " << g_size << ", VEC : " << vec_size << std::endl;

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

        std::cout << ">> ReorderKernelBlockedOpt::GetUpdateDispatchDataFunc : " <<kd.kernels[0].params.workGroups.global[0] << ", "
                    << kd.kernels[0].params.workGroups.global[1] << ", " << kd.kernels[0].params.workGroups.global[2] << std::endl;
    };
}

KernelsPriority ReorderKernelBlockedOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

static inline size_t SelectVecSizeFromSize(const DataTensor& tensor) {
    size_t size = tensor.PhysicalSize();
    auto preferred_vec_sizes = { 16, 8, 4, 2 };

    for (auto vec_size : preferred_vec_sizes) {
        if (size % vec_size == 0)
            return vec_size;
    }

    return 1;
}

static inline size_t SelectGroupSize(size_t ele_size) {
    // auto preferred_group_size = { 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2 };
    // auto preferred_group_size = { 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2 };
    auto preferred_group_size = { 32, 16, 8, 4, 2 };
    for (auto g_size : preferred_group_size) {
        if (ele_size % g_size == 0)
            return g_size;
    }

    return 1;
}
}  // namespace kernel_selector
