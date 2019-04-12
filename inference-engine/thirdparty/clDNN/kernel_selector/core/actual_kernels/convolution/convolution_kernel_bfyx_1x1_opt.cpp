/*
// Copyright (c) 2018 Intel Corporation
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

#include "convolution_kernel_bfyx_1x1_opt.h"

namespace kernel_selector 
{

    convolution_kernel_bfyx_1x1_opt::convolution_kernel_bfyx_1x1_opt() : ConvolutionKernelBase("convolution_gpu_bfyx_1x1_opt")
    {
    }

    ParamsKey convolution_kernel_bfyx_1x1_opt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableSubGroup();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        return k;
    }

    struct block_params
    {
        int32_t out_width;
        int32_t out_height;
        int32_t out_depth;
    };

    static block_params get_out_block_size(const convolution_params& p)
    {
        auto out_depth = 8;

        if (p.output.X().v == 7)
        {
            auto gws0 = p.output.X().v / 7;
            auto gws1 = p.output.Y().v / 1;
            auto gws2 = 2*(p.output.Feature().v * p.output.Batch().v) / 8 ; // process 8 output channels per Workitem

            auto compute_units = p.engineInfo.computeUnitsCount;
            auto total_threads = (gws0 * gws1 * gws2) / 64;
            if (total_threads < compute_units)
            {
                out_depth /= 2;
                total_threads *= 2;
            }
            if (total_threads < compute_units)
            {
                out_depth /= 2;
                total_threads *= 2;
            }
            return { 7,1,out_depth };
        }
        else if (p.output.X().v == 14)
            return { 7,1,8 };
        else if (p.output.X().v == 28)
            return { 7,2,4 };
        else if (p.output.X().v == 56)
            return { 8,1,8 };

        return { 1,1,1 };
    }


    ConvolutionKernelBase::DispatchData convolution_kernel_bfyx_1x1_opt::SetDefault(const convolution_params& cp, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(cp);

        constexpr size_t sub_group_size = 8;

        runInfo.effiency = FORCE_PRIORITY_3;

        auto block = get_out_block_size(cp);

        runInfo.gws0 = cp.output.X().v / block.out_width;
        runInfo.gws1 = cp.output.Y().v / block.out_height;
        runInfo.gws2 = 2*(cp.output.Feature().v * cp.output.Batch().v) / block.out_depth; // process 8 output channels per Workitem

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = 2*sub_group_size;

        return runInfo;
    }

    bool convolution_kernel_bfyx_1x1_opt::Validate(const Params& p, const optional_params& o) const
    {
        if (!ConvolutionKernelBase::Validate(p, o))
        {
            return false;
        }
        const convolution_params& cp = static_cast<const convolution_params&>(p);

        if (cp.stride.x != 1 || cp.stride.y != 1)
            return false;

        if (cp.filterSize.x != 1 || cp.filterSize.y != 1)
            return false;

        if (cp.output.Feature().v % 64 != 0)
            return false;

        if (cp.padding.x != 0 || cp.padding.y != 0)
            return false;

        // if block sizes are 1x1, then this algorithm is probably not the best
        auto block = get_out_block_size(cp);
        if (block.out_width == 1 && block.out_height == 1)
            return false;

        if (cp.output.X().v % block.out_width != 0)
            return false;
        if (cp.output.Y().v % block.out_height != 0)
            return false;

        return true;
    }

    JitConstants convolution_kernel_bfyx_1x1_opt::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        auto block = get_out_block_size(params);
        jit.AddConstant(MakeJitConstant("OUT_BLOCK_WIDTH", block.out_width));
        jit.AddConstant(MakeJitConstant("OUT_BLOCK_HEIGHT", block.out_height));
        jit.AddConstant(MakeJitConstant("OUT_BLOCK_DEPTH", block.out_depth));

        return jit;
    }

    std::vector<WeightsLayout> convolution_kernel_bfyx_1x1_opt::GetSupportedWeightLayouts(const convolution_params& cp) const
    {
        auto block = get_out_block_size(cp);
        if (block.out_depth == 8)
            return { WeightsLayout::os_iyx_osv64 };
        if (block.out_depth == 4)
            return { WeightsLayout::os_iyx_osv32 };
        if (block.out_depth == 2)
            return { WeightsLayout::os_iyx_osv16 };
        else
            return{ WeightsLayout::yxio };
    }

    KernelsData convolution_kernel_bfyx_1x1_opt::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options);
        if (!kd.empty())
            kd[0].estimatedTime = FORCE_PRIORITY_1;
        return kd;
    }

}