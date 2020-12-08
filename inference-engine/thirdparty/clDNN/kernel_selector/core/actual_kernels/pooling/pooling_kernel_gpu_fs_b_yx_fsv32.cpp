// Copyright (c) 2020 Intel Corporation
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


#include "pooling_kernel_gpu_fs_b_yx_fsv32.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
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
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableDifferentTypes();
    return k;
}

PoolingKernelBase::DispatchData PoolingKerneGPU_fs_b_yx_fsv32::SetDefault(const pooling_params& params) const {
    DispatchData dispatchData = PoolingKernelBase::SetDefault(params);

    dispatchData.gws[0] = params.output.X().v;  // X output blocks
    dispatchData.gws[1] = params.output.Y().v;  // Y output clocks
    // in fs_b_yx_fsv32 format we will process 2 features per work item, so reads/writes are done in full writes for
    // fp16
    dispatchData.gws[2] = RoundUp(params.output.Feature().v, 32) * params.output.Batch().v / 2;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 16;

    return dispatchData;
}

bool PoolingKerneGPU_fs_b_yx_fsv32::Validate(const Params& p, const optional_params& o) const {
    if (!PoolingKernelBase::Validate(p, o))
        return false;

    auto pp = static_cast<const pooling_params&>(p);

    // Feature padding before must be aligned to 32 to keep slices aligned
    if (pp.output.Feature().pad.before % 32 != 0)
        return false;

    if (pp.inputs[0].Feature().pad.before % 32 != 0)
        return false;

    return true;
}

JitConstants PoolingKerneGPU_fs_b_yx_fsv32::GetJitConstants(const pooling_params& params, DispatchData dispatchData) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, dispatchData);
    auto pp = static_cast<const pooling_params&>(params);

    // Heurestic needed for very big pool size.
    // ToDo Can it be changed to lower pool sizes?
    if (pp.poolSize.x >= 7 && pp.poolSize.y >= 7 && pp.poolType == PoolType::AVG) {
        jit.AddConstant(MakeJitConstant("USE_FLOAT_ACC", true));
    }
    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {"",
                                     {"b", "fs", "out_y", "out_x"},
                                     "pool_result",
                                     input_dt,
                                     2,
                                     LoadType::LT_ALIGNED_READ,
                                     BoundaryCheck::ENABLED,
                                     IndexType::TENSOR_COORD,
                                     Tensor::DataChannelName::FEATURE};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData PoolingKerneGPU_fs_b_yx_fsv32::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector
