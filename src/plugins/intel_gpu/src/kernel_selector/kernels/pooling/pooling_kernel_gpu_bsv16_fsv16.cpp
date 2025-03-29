// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_kernel_gpu_bsv16_fsv16.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;
static const size_t batch_block_size = 16;

ParamsKey PoolingKernel_bsv16_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey PoolingKernel_bsv16_fsv16::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

PoolingKernelBase::DispatchData PoolingKernel_bsv16_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(f, feature_block_size);
    dispatchData.gws[1] = x * y * z;
    dispatchData.gws[2] = CeilDiv(b, batch_block_size);

    dispatchData.lws[0] = sub_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority PoolingKernel_bsv16_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

bool PoolingKernel_bsv16_fsv16::Validate(const Params& p) const {
    if (!PoolingKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const pooling_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (output.Batch().v % batch_block_size != 0 || output.Feature().v % feature_block_size != 0)
        return false;

    if (input.Batch().v % batch_block_size != 0 || input.Feature().v % feature_block_size != 0)
        return false;

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0 ||
        input.Batch().pad.before % batch_block_size != 0 || output.Batch().pad.before % batch_block_size != 0) {
        return false;
    }

    return true;
}

JitConstants PoolingKernel_bsv16_fsv16::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto input = params.inputs[0];
    auto output = params.outputs[0];
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("OC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("MB_BLOCK", batch_block_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            idx_order = {"(b + BLOCK_NUM * 8)", "oc", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"(b + BLOCK_NUM * 8)", "oc", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"",
                                     idx_order,
                                     "pool_result",
                                     input_dt,
                                     8,
                                     LoadType::LT_ALIGNED_READ,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::BATCH};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData PoolingKernel_bsv16_fsv16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
