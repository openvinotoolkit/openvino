// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_opt.h"
#include <vector>
#include <core/common/kernel_selector_utils.h>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOpt::GetOptimalBlockSize(const resample_params& params) const {
    std::vector<size_t> block_width = { 16, 8, 4, 2, 1 };
    for (auto& w : block_width)
        if (params.output.X().v % w == 0)
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

ParamsKey ResampleKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableReampleType(ResampleType::LINEAR_ONNX);
    k.EnableReampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ResampleKernelBase::DispatchData ResampleKernelOpt::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    const auto& out = arg.output;

    if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        dispatchData.gws[0] = out.X().v * out.Y().v;
        dispatchData.gws[1] = CeilDiv(out.Feature().v, GetFeatureBlockSize(arg));
        dispatchData.gws[2] = arg.output.Batch().v;

        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo);
    } else {
        auto opt_x_block_size = GetOptimalBlockSize(arg);
        if (out.X().v > 32 && opt_x_block_size == 1) {
            opt_x_block_size = GetOptimalDivisor(out.X().v, 32);
        }

        dispatchData.gws[0] = CeilDiv(out.X().v, GetOptimalBlockSize(arg)) * out.Y().v;
        dispatchData.gws[1] = Align(out.Feature().v, sub_group_size);
        dispatchData.gws[2] = arg.output.Batch().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;

        if (arg.output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16
            || arg.output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32) {
            dispatchData.lws[2] = GetOptimalDivisor(dispatchData.gws[2]);
        }
    }

    return dispatchData;
}

KernelsPriority ResampleKernelOpt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_3;
}

bool ResampleKernelOpt::Validate(const Params& p, const optional_params& o) const {
    const resample_params& params = static_cast<const resample_params&>(p);

    if (!Parent::Validate(p, o))
        return false;

    if (p.GetType() != KernelType::RESAMPLE || o.GetType() != KernelType::RESAMPLE)
        return false;

    if (params.inputs.empty())
        return false;

    const auto& input = params.inputs[0];

    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::BILINEAR_INTERP)
        return false;

    if (input.GetLayout() != DataLayout::fs_b_yx_fsv32 &&
        input.GetLayout() != DataLayout::b_fs_yx_fsv16 &&
        input.GetLayout() != DataLayout::b_fs_yx_fsv32 &&
        input.GetLayout() != DataLayout::bs_fs_yx_bsv32_fsv16 &&
        input.GetLayout() != DataLayout::bs_fs_yx_bsv32_fsv32)
        return false;

    return true;
}

JitConstants ResampleKernelOpt::GetJitConstants(const resample_params &params) const {
    auto jit = Parent::GetJitConstants(params);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.output.X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.output.X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.output.X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

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
        if (params.resampleType != ResampleType::CAFFE_BILINEAR_INTERP) {
            std::vector<std::string> idx_order = {"b", "feature_block", "y", "(x + out_x)"};
            FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), vec_size, LoadType::LT_ALIGNED_READ};
            conf.SetVectorAxis(Tensor::DataChannelName::FEATURE);
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            std::vector<std::string> idx_order;
            idx_order = {"batch", "OF_ID", "oy", "ox"};

            FusedOpsConfiguration conf = {"", idx_order, "res", GetAccumulatorType(params), 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        if (GetFeatureBlockSize(params) == 8) {
            jit.AddConstant(MakeJitConstant("VEC_BLOCK_SIZE", 8));
        } else {
            jit.AddConstant(MakeJitConstant("VEC_BLOCK_SIZE", 16));
        }
    }

    return jit;
}

Datatype ResampleKernelOpt::GetUnitType(const base_params& params) const {
    return params.inputs[0].GetDType();
}

KernelsData ResampleKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
