// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_kernel_b_fs_zyx_fsv16.h"
#include "kernel_selector_utils.h"

#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;

ParamsKey DeconvolutionKernel_b_fs_zyx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey DeconvolutionKernel_b_fs_zyx_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_b_fs_zyx_fsv16::SetDefault(const deconvolution_params& params) const {
    DispatchData dispatchData = DeconvolutionKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    bool ver_bsv16_fsv16 = params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16
        || params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = Align(out.Feature().v, 16);
    auto b = out.Batch().v;

    if (ver_bsv16_fsv16) {
        dispatchData.gws[0] = 64;
        while (dispatchData.gws[0] > 16) {
            if (f % dispatchData.gws[0] == 0)
                break;
            dispatchData.gws[0] /= 2;
        }
        dispatchData.gws[1] = x * y * z;
        dispatchData.gws[2] = CeilDiv(b, 16) * (f / dispatchData.gws[0]) * params.groups;

        dispatchData.lws[0] = sub_group_size;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else {
        size_t x_block_size = 16;
        while (x_block_size > 1) {
            if (x % x_block_size == 0)
               break;
            x_block_size--;
        }
        x_block_size = std::max(x_block_size, (size_t)8);
        dispatchData.gws[0] = 64;
        while (dispatchData.gws[0] > 16) {
            if (f % dispatchData.gws[0] == 0)
                break;
            dispatchData.gws[0] /= 2;
        }
        dispatchData.gws[1] = CeilDiv(x, x_block_size) * y * z;
        dispatchData.gws[2] = b * (f / dispatchData.gws[0]);

        dispatchData.lws[0] = sub_group_size;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

KernelsPriority DeconvolutionKernel_b_fs_zyx_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool DeconvolutionKernel_b_fs_zyx_fsv16::Validate(const Params& p) const {
    if (!DeconvolutionKernelBase::Validate(p)) {
        return false;
    }
    auto& deconv_params = static_cast<const deconvolution_params&>(p);

    if (deconv_params.outputs[0].GetLayout() != deconv_params.inputs[0].GetLayout())
        return false;

    const auto& params = static_cast<const deconvolution_params&>(p);
    const auto feature_block_size = 16;

    // Check that padding features doesn't miss-align the blocks
    if (params.inputs[0].Feature().pad.before % feature_block_size != 0 || params.outputs[0].Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

JitConstants DeconvolutionKernel_b_fs_zyx_fsv16::GetJitConstants(const deconvolution_params& params) const {
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto jit = Parent::GetJitConstants(params);

    bool ver_bsv16_fsv16 = params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16
        || params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16;

    if (ver_bsv16_fsv16) {
        jit.AddConstant(MakeJitConstant("VER_16MB16C", 1));
    } else {
        jit.AddConstant(MakeJitConstant("VER_8OW16C", 1));
    }
    jit.AddConstant(MakeJitConstant("OC_BLOCK", 16));

    if (input.GetDType() == Datatype::F32) {
        jit.AddConstant(MakeJitConstant("DT_F32", 1));
    } else {
        jit.AddConstant(MakeJitConstant("DT_F16", 1));
    }

    auto mb_block = 1;
    auto ic_block = 16;
    auto iw_block = 1;
    auto icb = 64;
    while (icb > 16) {
        if (Align(output.Feature().v, 16) % icb == 0) break;
        icb /= 2;
    }

    if (ver_bsv16_fsv16) {
        mb_block = 16;
        jit.AddConstant(MakeJitConstant("MB_BLOCK", mb_block));
        jit.AddConstant(MakeJitConstant("IC_BLOCK", ic_block));
        jit.AddConstant(MakeJitConstant("IW_BLOCK", iw_block));
    } else {
        iw_block = 16;
        while (iw_block > 1) {
            if (output.X().v % iw_block == 0)
                break;
            iw_block--;
        }
        iw_block = std::max(iw_block, 8);
        jit.AddConstant(MakeJitConstant("MB_BLOCK", mb_block));
        jit.AddConstant(MakeJitConstant("IC_BLOCK", ic_block));
        jit.AddConstant(MakeJitConstant("IW_BLOCK", iw_block));
    }
    jit.AddConstant(MakeJitConstant("ICB", icb));
    jit.AddConstant(MakeJitConstant("IWB", CeilDiv(output.X().v, iw_block)));
    jit.AddConstant(MakeJitConstant("MB_LAST", (output.Batch().v / 16) * 16));
    jit.AddConstant(MakeJitConstant("G", 1));
    jit.AddConstant(MakeJitConstant("DD", params.dilation.z - 1));
    jit.AddConstant(MakeJitConstant("DH", params.dilation.y - 1));
    jit.AddConstant(MakeJitConstant("DW", params.dilation.x - 1));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("BWD_DATA", 1));
    jit.AddConstant(MakeJitConstant("WITH_BIAS", "BIAS_TERM"));

    jit.AddConstant(MakeJitConstant("MB", "OUTPUT_BATCH_NUM"));
    jit.AddConstant(MakeJitConstant("OC", Align(input.Feature().v, 16)));
    jit.AddConstant(MakeJitConstant("OD", "INPUT0_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("OH", "INPUT0_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("OW", "INPUT0_SIZE_X"));
    jit.AddConstant(MakeJitConstant("IC", Align(output.Feature().v, 16)));
    jit.AddConstant(MakeJitConstant("ID", "OUTPUT_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("IH", "OUTPUT_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("IW", "OUTPUT_SIZE_X"));
    jit.AddConstant(MakeJitConstant("KD", "FILTER_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("KH", "FILTER_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("KW", "FILTER_SIZE_X"));
    jit.AddConstant(MakeJitConstant("SD", "STRIDE_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("SH", "STRIDE_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("SW", "STRIDE_SIZE_X"));
    jit.AddConstant(MakeJitConstant("PD", "PADDING_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("PH", "PADDING_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("PW", "PADDING_SIZE_X"));

    jit.AddConstant(MakeJitConstant("OC_FULL", Align(params.inputs[0].Feature().LogicalDimPadded(), 16)));
    jit.AddConstant(MakeJitConstant("OD_FULL", params.inputs[0].Z().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("OH_FULL", params.inputs[0].Y().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("OW_FULL", params.inputs[0].X().LogicalDimPadded()));

    jit.AddConstant(MakeJitConstant("IC_FULL", Align(params.outputs[0].Feature().LogicalDimPadded(), 16)));
    jit.AddConstant(MakeJitConstant("ID_FULL", params.outputs[0].Z().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("IH_FULL", params.outputs[0].Y().LogicalDimPadded()));
    jit.AddConstant(MakeJitConstant("IW_FULL", params.outputs[0].X().LogicalDimPadded()));


    DispatchData dispatchData = SetDefault(params);
    jit.AddConstant(MakeJitConstant("LWS_0", dispatchData.lws[0]));
    jit.AddConstant(MakeJitConstant("LWS_1", dispatchData.lws[1]));
    jit.AddConstant(MakeJitConstant("LWS_2", dispatchData.lws[2]));

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        std::vector<std::string> idx_order_block_c00;
        std::vector<std::string> idx_order_block_c01;
        std::vector<std::string> idx_order_block_ci;

        if (params.outputs[0].Dimentions() <= 4) {
            idx_order_block_c00 = { "mb", "(g * IC + gic * IC_BLOCK)", "ih", "iw" };
            idx_order_block_c01 = { "(mb + 8)", "(g * IC + gic * IC_BLOCK)", "ih", "iw" };
            idx_order_block_ci = { "mb", "(g * IC + gic * IC_BLOCK)", "ih", "(iw + i)" };
        } else {
            idx_order_block_c00 = { "mb", "(g * IC + gic * IC_BLOCK)", "id", "ih", "iw" };
            idx_order_block_c01 = { "(mb + 8)", "(g * IC + gic * IC_BLOCK)", "id", "ih", "iw" };
            idx_order_block_ci = { "mb", "(g * IC + gic * IC_BLOCK)", "id", "ih", "(iw + i)" };
        }

        FusedOpsConfiguration conf_c00 = {
            "_BLOCK_C00",
            idx_order_block_c00,
            "blockC00",
            fused_dt,
            8,
            LoadType::LT_ALIGNED_READ,
            BoundaryCheck::ENABLED,
            IndexType::TENSOR_COORD,
            Tensor::DataChannelName::BATCH };
        FusedOpsConfiguration conf_c01 = {
            "_BLOCK_C01",
            idx_order_block_c01,
            "blockC01",
            fused_dt,
            8,
            LoadType::LT_ALIGNED_READ,
            BoundaryCheck::ENABLED,
            IndexType::TENSOR_COORD,
            Tensor::DataChannelName::BATCH };

        auto load_type = LoadType::LT_ALIGNED_READ;
        for (auto& fused_op : params.fused_ops) {
            if (!fused_op.output_tensor.SameDims(params.outputs[0]) &&
                (fused_op.output_tensor.X().v > 1 || fused_op.output_tensor.Y().v > 1 || fused_op.output_tensor.Z().v > 1)) {
                load_type = LoadType::LT_UNALIGNED;
                idx_order_block_ci[1] = "(g * IC + gic * IC_BLOCK + local_id)";
            }
        }
        FusedOpsConfiguration conf_ci = { "_BLOCK_CI", idx_order_block_ci, "blockC00[i]", fused_dt, 1, load_type };

        jit.Merge(MakeFusedOpsJitConstants(params, { conf_c00, conf_c01, conf_ci }));
    }

    return jit;
}

KernelsData DeconvolutionKernel_b_fs_zyx_fsv16::GetKernelsData(const Params& params) const {
    KernelsData kds = Parent::GetKernelsData(params);

    const deconvolution_params& orgParams = static_cast<const deconvolution_params&>(params);
    if (!kds.empty() && orgParams.inputs[0].Feature().v % 16 != 0) {
        kds[0].can_reuse_memory = false; // Set memory_reuse = false when input feature size is not 16 aligned.
    }

    return kds;
}
}  // namespace kernel_selector
