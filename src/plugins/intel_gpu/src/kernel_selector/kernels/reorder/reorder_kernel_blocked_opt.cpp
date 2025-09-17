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
    // k.EnableSurfaceInputSupport();
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

ReorderKernelBase::DispatchData ReorderKernelBlockedOpt::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;
    size_t global_w_item = std::max(params.outputs[0].PhysicalSize() / SelectVecSizeFromSize(params.outputs[0]), (size_t)1);
    size_t g_size = SelectGroupSize(global_w_item);

    //  dispatchData.gws = {global_w_item, 1, 1};
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
    // if (p.layerID == "result:Result_33640" || params.outputs[0].PhysicalSize() > 10000) {
    //     GPU_DEBUG_COUT << "  >>" << p.layerID << std::endl;
    // }

    if (SelectVecSizeFromSize(params.outputs[0]) == 1)
        return false;

    if (!params.fused_ops.empty())
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
    if (input.GetDims().size() != output.GetDims().size() || !compare_tensors(input, output)) {
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

    if (params.surface_input)
        jit.AddConstant(MakeJitConstant("SURFACE_INPUT", true));

    // if (!params.fused_ops.empty()) {
    //     std::vector<std::string> idx_order;
    //     if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
    //         idx_order = {"b", "f", "y", "x"};
    //     } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
    //         idx_order = {"b", "f", "z", "y", "x"};
    //     }
    //     FusedOpsConfiguration conf = {"", idx_order, "res", GetUnitType(params), 1};
    //     jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    // }

    size_t vec_size = SelectVecSizeFromSize(params.outputs[0]);
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    size_t global_w_item = std::max(params.outputs[0].PhysicalSize() / vec_size, (size_t)1);
    size_t g_size = SelectGroupSize(global_w_item);
    jit.AddConstant(MakeJitConstant("ITEM_SIZE", g_size));
    GPU_DEBUG_COUT << ">> " << params.layerID << " >> ITEM_SIZE : " << g_size << ", VEC : " << vec_size << std::endl;

    // if ( params.inputs[0].GetDType() == Datatype::BF16 ) {
    //      jit.AddConstant(MakeJitConstant("BF16_INPUT", true));
    // }

    return jit;
}

KernelsData ReorderKernelBlockedOpt::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
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
