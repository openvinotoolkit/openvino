// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static inline int GetInnerFeatureBlockSize(const DataTensor&);

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

    size_t feature_block_size = GetInnerFeatureBlockSize(params.outputs[0]);
    jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));

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

static inline int GetInnerFeatureBlockSize(const DataTensor& tensor) {
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::b_fs_yx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 16;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
        return 32;
    case DataLayout::bfyx:
    case DataLayout::bfzyx:
    default:
        return 1;
    }

    return 1;
}
}  // namespace kernel_selector
