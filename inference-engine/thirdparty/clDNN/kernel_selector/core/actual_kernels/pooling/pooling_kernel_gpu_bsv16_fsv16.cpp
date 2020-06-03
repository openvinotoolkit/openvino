// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

PoolingKernelBase::DispatchData PoolingKernel_bsv16_fsv16::SetDefault(const pooling_params& params) const {
    DispatchData kd = PoolingKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = Align(f, feature_block_size);
    kd.gws1 = x * y * z;
    kd.gws2 = CeilDiv(b, batch_block_size);

    kd.lws0 = sub_group_size;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.efficiency = FORCE_PRIORITY_1;

    return kd;
}

bool PoolingKernel_bsv16_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!PoolingKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const pooling_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

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

JitConstants PoolingKernel_bsv16_fsv16::GetJitConstants(const pooling_params& params, DispatchData runInfo) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = PoolingKernelBase::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("OC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("MB_BLOCK", batch_block_size));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", feature_block_size));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"(b + BLOCK_NUM * 8)", "oc", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
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

KernelsData PoolingKernel_bsv16_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector
