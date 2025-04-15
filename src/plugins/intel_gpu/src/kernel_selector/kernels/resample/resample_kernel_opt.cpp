// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_opt.h"
#include <vector>
#include <kernel_selector_utils.h>

namespace kernel_selector {

static constexpr size_t sub_group_size = 16;

size_t ResampleKernelOpt::GetOptimalBlockSize(const resample_params& params) const {
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
    k.EnableReampleType(ResampleType::BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableReampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    return k;
}

DeviceFeaturesKey ResampleKernelOpt::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

static size_t get_vec_size(const resample_params &params) {
    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        return 2;
    } else {
        return 1;
    }
}

static int get_feature_slice_size(const resample_params &params) {
    return static_cast<int>(16 * get_vec_size(params));
}

ResampleKernelBase::DispatchData ResampleKernelOpt::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    size_t dims = arg.outputs[0].Dimentions();
    const auto& out = arg.outputs[0];

    if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        dispatchData.gws[0] = out.X().v * out.Y().v;
        dispatchData.gws[1] = CeilDiv(out.Feature().v, GetFeatureBlockSize(arg));
        dispatchData.gws[2] = arg.outputs[0].Batch().v;

        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE},
                       {Tensor::DataChannelName::BATCH}};
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);
    } else {
        auto opt_x_block_size = GetOptimalBlockSize(arg);
        if (out.X().v > 32 && opt_x_block_size == 1) {
            opt_x_block_size = GetOptimalDivisor(out.X().v, 32);
        }

        if (dims == 5) {
            dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v * out.Z().v;
        } else {
            dispatchData.gws[0] = CeilDiv(out.X().v, opt_x_block_size) * out.Y().v;
        }
        dispatchData.gws[1] = Align(CeilDiv(out.Feature().v, get_vec_size(arg)), sub_group_size);
        dispatchData.gws[2] = arg.outputs[0].Batch().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;

        if (arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
            arg.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
            dispatchData.lws[2] = GetOptimalDivisor(dispatchData.gws[2]);
        }
    }

    return dispatchData;
}

KernelsPriority ResampleKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

bool ResampleKernelOpt::Validate(const Params& p) const {
    const resample_params& params = static_cast<const resample_params&>(p);
    if (!Parent::Validate(p))
        return false;

    const auto& input = params.inputs[0];

    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::BILINEAR_INTERP)
        return false;

    // in the case of 5D support only NEAREST_NEIGHBOR
    if (input.Dimentions() == 5 && params.resampleType != ResampleType::NEAREST_NEIGHBOR)
        return false;

    return true;
}

JitConstants ResampleKernelOpt::GetJitConstants(const resample_params &params) const {
    auto jit = Parent::GetJitConstants(params);

    auto opt_x_block_size = GetOptimalBlockSize(params);
    if (params.outputs[0].X().v > 32 && opt_x_block_size == 1) {
        opt_x_block_size = GetOptimalDivisor(params.outputs[0].X().v, 32);
    }

    jit.AddConstant(MakeJitConstant("OUTPUT_X_BLOCK_SIZE", opt_x_block_size));
    jit.AddConstant(MakeJitConstant("X_BLOCKS", CeilDiv(params.outputs[0].X().v, opt_x_block_size)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));

    const size_t vec_size = get_vec_size(params);
    jit.AddConstant(MakeJitConstant("FEATURE_SLICE_SIZE", get_feature_slice_size(params)));
    jit.AddConstant(MakeJitConstant("VEC_SIZE", vec_size));

    if (!params.fused_ops.empty()) {
        if (params.resampleType != ResampleType::CAFFE_BILINEAR_INTERP) {
            std::vector<std::string> idx_order;
            if (params.inputs[0].Dimentions() == 5)
                idx_order = {"b", "feature_block", "z", "y", "(x + out_x)"};
            else
                idx_order = {"b", "feature_block", "y", "(x + out_x)"};
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

KernelsData ResampleKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
