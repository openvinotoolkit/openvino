// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
namespace {
size_t GetSingleFeatureBlockSize(DataLayout layout) {
    switch (layout) {
        case DataLayout::b_fs_yx_fsv2:
        case DataLayout::b_fs_zyx_fsv2:
            return 2;
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
    const auto feature_block_size = GetSingleFeatureBlockSize(output.GetLayout());
    return !params.is_shape_agnostic &&
           params.operations.size() == 1 &&
           params.operations[0].mode == EltwiseMode::ASSIGN &&
           feature_block_size != 0 &&
           !output.Feature().is_dynamic &&
           !output.Feature().pad.is_dynamic &&
           output.Feature().pad.Total() == 0 &&
           output.Feature().v % feature_block_size != 0;
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

JitConstants EltwiseKernelRef::GetJitConstants(const eltwise_params& params) const {
    auto jit = EltwiseKernelBase::GetJitConstants(params);

    if (ShouldZeroOutputFeaturePadding(params)) {
        const auto& output = params.outputs[0];
        const auto aligned_feature_size = Align(output.Feature().v, GetSingleFeatureBlockSize(output.GetLayout()));
        const auto feature_channel_idx = DataTensor::Channelndex(output.GetLayout(), Tensor::DataChannelName::FEATURE);
        jit.AddConstant(MakeJitConstant("ZERO_OUTPUT_FEATURE_PADDING", 1));
        jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_GWS_SIZE", aligned_feature_size));
        jit.AddConstant(MakeJitConstant("OUTPUT_FEATURE_INDEX", "d" + std::to_string(feature_channel_idx + 1)));
    }

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            if (!params.layoutBased && !params.int8_quantization && !params.broadcast && !CheckInputsOutputNoPitchSameDims(params)) {
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

        if (!params.layoutBased && !params.int8_quantization && !params.broadcast && CheckInputsOutputNoPitchSameDims(params)) {
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
    const auto aligned_feature_size = Align(output.Feature().v, GetSingleFeatureBlockSize(output.GetLayout()));
    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        dispatch_data.gws[1] = aligned_feature_size;
    } else {
        dispatch_data.gws[2] = aligned_feature_size * output.Batch().v;
    }
}
}  // namespace kernel_selector
