// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_onnx_5d.h"
#include <vector>
#include <kernel_selector_utils.h>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOnnx5d::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = { 16, 8, 4, 2, 1 };
    for (auto& w : block_width)
        if (params.outputs[0].X().v % w == 0)
            return w;
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

ParamsKey ResampleKernelOnnx5d::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);

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
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ResampleKernelBase::DispatchData ResampleKernelOnnx5d::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = arg.outputs[0];

    auto opt_x_block_size = GetOptimalBlockSize(arg);
    if (out.X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(out.X().v, 32);
    }

    dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v * out.Z().v;
    dispatchData.gws[1] = Align(out.Feature().v, sub_group_size);
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

KernelsPriority ResampleKernelOnnx5d::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_4;
}

bool ResampleKernelOnnx5d::Validate(const Params& p, const optional_params& o) const {
    const resample_params& params = static_cast<const resample_params&>(p);
    if (!Parent::Validate(p, o))
        return false;

    if (p.GetType() != KernelType::RESAMPLE || o.GetType() != KernelType::RESAMPLE)
        return false;

    if (params.inputs.empty())
        return false;

    const auto& input = params.inputs[0];

    if (input.Dimentions() != 5 || params.resampleType != ResampleType::LINEAR_ONNX)
        return false;

    return true;
}

JitConstants ResampleKernelOnnx5d::GetJitConstants(const resample_params &params) const {
    auto jit = Parent::GetJitConstants(params);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.outputs[0].X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.outputs[0].X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

    // TODO consider remove it
    size_t vec_size = 0;
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        vec_size = 2;
        jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", 32));
    } else {
        vec_size = 1;
        jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", 16));
    }
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        idx_order = {"batch", "OF_ID", "oy", "ox"};

        FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

Datatype ResampleKernelOnnx5d::GetUnitType(const base_params& params) const {
    return params.inputs[0].GetDType();
}

KernelsData ResampleKernelOnnx5d::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
