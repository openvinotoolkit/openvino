// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_onnx.h"

#include <kernel_selector_utils.h>

#include <vector>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOnnx::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = {16, 8, 4, 2, 1};
    for (auto& w : block_width) {
        if (params.outputs[0].X().v % w == 0) {
            return w;
        }
    }
    return 1;
}

static size_t GetOptimalDivisor(const size_t input_size, size_t max_val = 16) {
    for (size_t s = max_val; s > 0; --s) {
        if (input_size % s == 0) {
            return s;
        }
    }
    return 1;
}

ParamsKey ResampleKernelOnnx::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);

    // 4d formats
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);

    // 5d formats
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv32_fsv32);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::LINEAR_ONNX);
    return k;
}

DeviceFeaturesKey ResampleKernelOnnx::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

static size_t get_vec_size(const resample_params &params) {
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        return 2;
    } else {
        return 1;
    }
}

ResampleKernelBase::DispatchData ResampleKernelOnnx::SetDefault(const kernel_selector::resample_params& arg) const {
    DispatchData dispatchData;
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = arg.outputs[0];

    auto opt_x_block_size = GetOptimalBlockSize(arg);
    if (out.X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(out.X().v, 32);
    }

    dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v * out.Z().v;
    dispatchData.gws[1] = Align(CeilDiv(out.Feature().v, get_vec_size(arg)), sub_group_size);
    dispatchData.gws[2] = arg.outputs[0].Batch().v;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = sub_group_size;
    dispatchData.lws[2] = 1;

    if (arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
        arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        dispatchData.lws[2] = GetOptimalDivisor(dispatchData.gws[2]);
    }

    return dispatchData;
}

KernelsPriority ResampleKernelOnnx::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}

bool ResampleKernelOnnx::Validate(const Params& p) const {
    const resample_params& params = static_cast<const resample_params&>(p);

    if (!Parent::Validate(p))
        return false;

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    if (input.Batch().v != output.Batch().v || input.Feature().v != output.Feature().v)
        return false;

    return true;
}

static bool IsThreeSpatialResample(const resample_params& params) {
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (input.Dimentions() == 5 && input.Z().v != output.Z().v)
        return true;

    return false;
}

JitConstants ResampleKernelOnnx::GetJitConstants(const resample_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.outputs[0].X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.outputs[0].X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

    size_t vec_size = get_vec_size(params);
    jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", 16 * vec_size));

    if (IsThreeSpatialResample(params))
        jit.AddConstant(MakeJitConstant("THREE_SPATIAL_RESAMPLE", ""));

    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].Dimentions() == 5)
            idx_order = {"b", "feature_block", "z", "y", "(x + out_x)"};
        else
            idx_order = {"b", "feature_block", "y", "(x + out_x)"};
        FusedOpsConfiguration conf =
            {"", idx_order, "res", GetAccumulatorType(params), vec_size, LoadType::LT_ALIGNED_READ};
        conf.SetVectorAxis(Tensor::DataChannelName::FEATURE);
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData ResampleKernelOnnx::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
