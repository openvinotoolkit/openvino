// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey ActivationKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableBatching();
    return k;
}

static size_t GetTotalSize(const activation_params& params) {
    const auto input = params.inputs[0];
    size_t totalSize = input.LogicalSize();
    switch (params.inputs[0].GetLayout()) {
        case DataLayout::b_fs_yx_fsv4:
            totalSize = (totalSize / input.Feature().v) * Align(input.Feature().v, 4);
            break;
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::b_fs_zyx_fsv16:
            totalSize = (totalSize / input.Feature().v) * Align(input.Feature().v, 16);
            break;
        case DataLayout::b_fs_yx_fsv32:
        case DataLayout::b_fs_zyx_fsv32:
        case DataLayout::fs_b_yx_fsv32:
            totalSize = (totalSize / input.Feature().v) * Align(input.Feature().v, 32);
            break;
        case DataLayout::bs_fs_zyx_bsv16_fsv16:
        case DataLayout::bs_fs_yx_bsv16_fsv16:
            totalSize = (totalSize / (input.Feature().v * input.Batch().v)) * Align(input.Feature().v, 16) * Align(input.Batch().v, 16);
            break;
        default: break;
    }
    return totalSize;
}

ActivationKernelOpt::Parent::DispatchData ActivationKernelOpt::SetDefault(const activation_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    dispatchData.gws = { GetTotalSize(params) / NUM_COLS_WI, 1, 1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority ActivationKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}

bool ActivationKernelOpt::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ACTIVATION) {
        return false;
    }

    const activation_params& params = static_cast<const activation_params&>(p);

    const auto totalSize = GetTotalSize(params);
    if ((totalSize % NUM_COLS_WI) != 0 ||
        (params.inputs[0].GetFirstElementOffset() % NUM_COLS_WI) != 0 ||
        (params.outputs[0].GetFirstElementOffset() % NUM_COLS_WI) != 0) {
        return false;
    }

    if (params.outputs[0].GetDims().size() > 5)
        return false;

    if (params.outputs[0].GetLayout() != params.inputs[0].GetLayout())
        return false;

    if (!params.fused_ops.empty() &&
        (params.outputs[0].GetLayout() != DataLayout::bfyx && params.outputs[0].GetLayout() != DataLayout::bfzyx))
        return false;

    auto input_dt = params.inputs[0].GetDType();
    if (input_dt == Datatype::INT8 || input_dt == Datatype::INT32) {
        for (auto act : params.activations) {
            if (act.function == ActivationFunction::ABS)
                return false;
        }
    }

    return true;
}

JitConstants ActivationKernelOpt::GetJitConstants(const activation_params& params, DispatchData dispatchData) const {
    auto jit = ActivationKernelBase::GetJitConstants(params, dispatchData);
    auto input_dt = params.inputs[0].GetDType();

    jit.AddConstant(MakeJitConstant("NUM_COLS_WI", NUM_COLS_WI));
    if (!params.fused_ops.empty()) {
        bool can_use_vector = params.inputs[0].X().v % 4 == 0;
        jit.AddConstant(MakeJitConstant("CAN_USE_VECTOR", can_use_vector));

        std::vector<std::string> idx_order;

        if (can_use_vector) {
            if (params.inputs[0].GetDims().size() <= 4) {
                idx_order = {"x / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_FEATURE_NUM)",
                             "x / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM",
                             "x / OUTPUT_SIZE_X % OUTPUT_SIZE_Y",
                             "x % OUTPUT_SIZE_X"};
            } else if (params.inputs[0].GetDims().size() == 5) {
                idx_order = {"x / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z* OUTPUT_FEATURE_NUM)",
                             "x / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z) % OUTPUT_FEATURE_NUM",
                             "x / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z",
                             "x / OUTPUT_SIZE_X % OUTPUT_SIZE_Y",
                             "x % OUTPUT_SIZE_X"};
            }
        } else {
            if (params.inputs[0].GetDims().size() <= 4) {
                idx_order = {"(x + i) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_FEATURE_NUM)",
                             "(x + i) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_FEATURE_NUM",
                             "(x + i) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y",
                             "(x + i) % OUTPUT_SIZE_X"};
            } else if (params.inputs[0].GetDims().size() == 5) {
                idx_order = {"(x + i) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z* OUTPUT_FEATURE_NUM)",
                             "(x + i) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z) % OUTPUT_FEATURE_NUM",
                             "(x + i) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y) % OUTPUT_SIZE_Z",
                             "(x + i) / OUTPUT_SIZE_X % OUTPUT_SIZE_Y",
                             "(x + i) % OUTPUT_SIZE_X"};
            }
        }
        FusedOpsConfiguration conf_vector = {"_VECTOR",
                                             idx_order,
                                             "v",
                                             input_dt,
                                             4,
                                             LoadType::LT_UNALIGNED,
                                             BoundaryCheck::DISABLED,
                                             IndexType::TENSOR_COORD,
                                             Tensor::DataChannelName::X};
        FusedOpsConfiguration conf_scalar = {"_SCALAR",
                                             idx_order,
                                             "v[i]",
                                             input_dt,
                                             1,
                                             LoadType::LT_UNALIGNED,
                                             BoundaryCheck::DISABLED,
                                             IndexType::TENSOR_COORD};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vector, conf_scalar}));
    }
    jit.Merge(MakeActivationJitConstants(params.activations, input_dt, "_KERNEL"));

    return jit;
}

KernelsData ActivationKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
