/*
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
*/

#include "pooling_kernel_gpu_bfyx_block_opt.h"
 
namespace kernel_selector 
{
    ParamsKey PoolingKernelGPUBfyxBlockOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        k.EnablePoolType(PoolType::MAX);
        k.EnablePoolType(PoolType::AVG);
        k.EnablePoolType(PoolType::MAX_WITH_ARGMAX);
        k.EnablePoolRemainder(PoolRemainder::FLOOR);
        k.EnablePoolRemainder(PoolRemainder::CEIL);
        k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
        k.EnableDifferentTypes();
        return k;
    }

    PoolingKernelBase::DispatchData PoolingKernelGPUBfyxBlockOpt::SetDefault(const pooling_params& params) const
    {
        const auto& output = params.output;

        DispatchData runInfo = PoolingKernelBase::SetDefault(params);

        runInfo.gws1 = CeilDiv(output.Y().v, params.poolSize.y);

        return runInfo;
    }

    JitConstants PoolingKernelGPUBfyxBlockOpt::GetJitConstants(const pooling_params& params, DispatchData kd) const
    {
        auto mem_consts = PoolingKernelBase::GetJitConstants(params, kd);

        mem_consts.AddConstant(MakeJitConstant("BLOCK_SIZE_Y", params.poolSize.y + params.poolSize.y*params.poolStride.y - 1));

        return mem_consts;
    }

    bool PoolingKernelGPUBfyxBlockOpt::Validate(const Params& p, const optional_params& o) const
    {
        if (!PoolingKernelBase::Validate(p, o))
        {
            return false;
        }

        const pooling_params& params = static_cast<const pooling_params&>(p);
        if (NeedsBoundaryCheck(params) ||
            params.poolSize.x > 5 || params.poolSize.y > 5)
        {
            return false;
        }

        return true;
    }

    KernelsData PoolingKernelGPUBfyxBlockOpt::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options, FORCE_PRIORITY_8);
    }
}