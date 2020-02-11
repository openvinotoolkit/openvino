// Copyright (c) 2016 Intel Corporation
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


#include "pooling_kernel_gpu_int8_ref.h"
#include <string>

namespace kernel_selector {
ParamsKey PoolingKernelGPUInt8Ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
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

KernelsData PoolingKernelGPUInt8Ref::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
}

JitConstants PoolingKernelGPUInt8Ref::GetJitConstants(const pooling_params& params, DispatchData kd) const {
    JitConstants jit = PoolingKernelBase::GetJitConstants(params, kd);

    if (!params.fused_ops.empty()) {
        auto input_dt = EnableRound(params) ? Datatype::INT32 : GetActivationType(params);
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"b", "f", "y", "x"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"b", "f", "z", "y", "x"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "pool_res", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool PoolingKernelGPUInt8Ref::Validate(const Params& params, const optional_params& options) const {
    if (!PoolingKernelBase::Validate(params, options)) {
        return false;
    }
    auto p = dynamic_cast<const pooling_params&>(params);

    if (p.inputs[0].GetDType() == Datatype::INT8 || p.inputs[0].GetDType() == Datatype::UINT8) {
        // Max pooling doesn't change quantization ranges, so output data type should be the same as input
        if ((p.poolType == PoolType::MAX || p.poolType == PoolType::MAX_WITH_ARGMAX) && p.output.GetDType() != p.inputs[0].GetDType())
            return false;
//         Average pooling should produce FP by default. (u)int8 is possible when quantize op is fused.
//        if (p.poolType == PoolType::AVG &&
//            !((p.output.GetDType() == p.inputs[0].GetDType() && !p.fused_ops.empty()) ||
//              (p.output.GetDType() == Datatype::F32 || p.output.GetDType() == Datatype::F16)))
//            return false;
    }

    return true;
}

}  // namespace kernel_selector
