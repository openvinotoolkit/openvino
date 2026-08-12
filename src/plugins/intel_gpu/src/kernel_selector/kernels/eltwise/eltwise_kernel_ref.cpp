// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

#include <limits>

namespace kernel_selector {
namespace {
size_t GetFeatureBlockSize(DataLayout layout) {
    switch (layout) {
        case DataLayout::b_fs_yx_fsv4:
        case DataLayout::b_fs_zyx_fsv4:
            return 4;
        case DataLayout::b_fs_yx_fsv8:
        case DataLayout::b_fs_zyx_fsv8:
            return 8;
        case DataLayout::b_fs_yx_fsv16:
        case DataLayout::b_fs_zyx_fsv16:
            return 16;
        case DataLayout::b_fs_yx_fsv32:
        case DataLayout::b_fs_zyx_fsv32:
            return 32;
        default:
            return 0;
    }
}

bool ShouldZeroOutputFeaturePadding(const eltwise_params& params) {
    const auto& output = params.outputs[0];
    const auto& feature = output.Feature();
    const auto feature_block_size = GetFeatureBlockSize(output.GetLayout());
    return feature_block_size != 0 &&
           !feature.pad.is_dynamic &&
           (params.is_shape_agnostic || feature.LogicalDimPadded() % feature_block_size != 0);
}
}  // namespace

ParamsKey EltwiseKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT16);
    k.EnableInputDataType(Datatype::UINT16);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT16);
    k.EnableOutputDataType(Datatype::UINT16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableEltwiseStride();
    k.EnableEltwiseBroadcast();
    k.EnableDynamicShapesSupport();
    return k;
}

bool EltwiseKernelRef::Validate(const Params& p) const {
    if (!EltwiseKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

KernelsData EltwiseKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority EltwiseKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants EltwiseKernelRef::MakeIndexJitConstants(const eltwise_params& params, bool use_vload8) const {
    if (!ShouldZeroOutputFeaturePadding(params) ||
        params.layoutBased ||
        params.int8_quantization ||
        params.broadcast ||
        !CheckInputsOutputNoPitchSameDims(params)) {
        return EltwiseKernelBase::MakeIndexJitConstants(params, use_vload8);
    }

    auto non_linear_params = params;
    non_linear_params.layoutBased = true;
    return EltwiseKernelBase::MakeIndexJitConstants(non_linear_params, use_vload8);
}

JitConstants EltwiseKernelRef::GetJitConstants(const eltwise_params& params) const {
    auto jit = EltwiseKernelBase::GetJitConstants(params);

    const auto& output = params.outputs[0];
    const auto feature_block_size = GetFeatureBlockSize(output.GetLayout());
    const bool zero_output_feature_padding = ShouldZeroOutputFeaturePadding(params);
    if (zero_output_feature_padding) {
        const std::string block_size = std::to_string(feature_block_size);
        const std::string padded_feature_size =
            "(OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM)";
        const std::string feature_gws_size =
            "(OUTPUT_FEATURE_NUM + (" + block_size + " - (" + padded_feature_size + " % " + block_size + ")) % " +
            block_size + ")";
        jit.RemoveConstant("ELTWISE_NO_PITCH_SAME_DIMS");
        jit.AddConstant(MakeJitConstant("ELTWISE_NO_PITCH_SAME_DIMS", 0));
        jit.AddConstant(MakeJitConstant("ZERO_OUTPUT_FEATURE_PADDING", 1));
        jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_GWS_SIZE", feature_gws_size));
    }

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);
        const bool no_pitch_same_dims = !zero_output_feature_padding && CheckInputsOutputNoPitchSameDims(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            if (!params.layoutBased && !params.int8_quantization && !params.broadcast && !no_pitch_same_dims) {
                auto calc_dim = [&params](Tensor::DataChannelName channel) {
                    int idx = DataTensor::Channelndex(params.outputs[0].GetLayout(), channel);
                    // We increment the index, because fusions dims ordering starts from one
                    return "d" + std::to_string(idx + 1);
                };

                idx_order = {calc_dim(Tensor::DataChannelName::BATCH),
                             calc_dim(Tensor::DataChannelName::FEATURE),
                             calc_dim(Tensor::DataChannelName::Y),
                             calc_dim(Tensor::DataChannelName::X)};
            } else {
                idx_order = {"d4", "d3", "d2", "d1"};
            }
        } else {
            size_t channels = DataTensor::ChannelsCount(params.outputs[0].GetLayout());
            idx_order.resize(channels);
            for (size_t i = 0; i < channels; i++) {
                idx_order[i] = "d" + std::to_string(channels - i);
            }
        }

        if (!params.layoutBased && !params.int8_quantization && !params.broadcast && no_pitch_same_dims) {
            FusedOpsConfiguration conf = {"", {"d1"}, "res", input_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::ENABLED, IndexType::LINEAR_OFFSET};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            FusedOpsConfiguration conf =  {"", idx_order, "res", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    return jit;
}

void EltwiseKernelRef::AdjustGlobalWorkSizes(const eltwise_params& params, DispatchData& dispatch_data) const {
    if (!ShouldZeroOutputFeaturePadding(params))
        return;

    const auto& output = params.outputs[0];
    if (output.Feature().is_dynamic || output.Feature().pad.is_dynamic)
        return;

    const auto feature_block_size = GetFeatureBlockSize(output.GetLayout());
    const auto padded_feature_size = output.Feature().LogicalDimPadded();
    const auto remainder = padded_feature_size % feature_block_size;
    const auto feature_tail_size = remainder == 0 ? 0 : feature_block_size - remainder;
    OPENVINO_ASSERT(output.Feature().v <= std::numeric_limits<size_t>::max() - feature_tail_size,
                    "Eltwise feature global work size overflow");
    const auto f_gws_size = output.Feature().v + feature_tail_size;
    if (f_gws_size == 0)
        return;

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        dispatch_data.gws[1] = f_gws_size;
    } else {
        const auto& dims = output.GetDims();
        dispatch_data.gws[0] = dims[0].v;
        dispatch_data.gws[1] = dims.size() == 5 ? dims[1].v * dims[2].v : dims[1].v;
        OPENVINO_ASSERT(output.Batch().v <= std::numeric_limits<size_t>::max() / f_gws_size,
                        "Eltwise global work size overflow");
        dispatch_data.gws[2] = f_gws_size * output.Batch().v;
    }
}
}  // namespace kernel_selector
