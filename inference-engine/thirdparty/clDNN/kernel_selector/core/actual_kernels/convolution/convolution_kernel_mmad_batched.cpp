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

#include "convolution_kernel_mmad_batched.h"

namespace kernel_selector {
    
    ParamsKey ConvolutionKernel_mmad_batched::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
        k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        k.DisableTuning();
        return k;
    }

    ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_batched::SetDefault(const convolution_params& arg, int) const
    {
        DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

        constexpr size_t sub_group_size = 8;

        const auto of_maps = arg.output.Feature().v;
        const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

        runInfo.effiency = FORCE_PRIORITY_6;

        runInfo.gws0 = arg.output.X().v;
        runInfo.gws1 = arg.output.Y().v;
        runInfo.gws2 = of_threads_per_batch * ((arg.output.Batch().v+3) / 4);

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = sub_group_size;

        return runInfo;
    }

    JitConstants ConvolutionKernel_mmad_batched::GetJitConstants(const convolution_params& params, const DispatchData& runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);

        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));

        // pitch for special block format used in this kernel
        const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
        const size_t filter_ofm_block_pitch = (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
        jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));

        const size_t in_x_pitch = 32 * 4;
        const size_t in_y_pitch = 32 * 4 * params.inputs[0].X().LogicalDimPadded();
        const size_t in_b_block_pitch = in_y_pitch * params.inputs[0].Y().LogicalDimPadded();
        const size_t in_f_block_pitch = in_b_block_pitch * ((params.inputs[0].Batch().v + 3) / 4);
        const size_t in_offset = in_x_pitch * params.inputs[0].X().pad.before + in_y_pitch * params.inputs[0].Y().pad.before;

        jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
        jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
        jit.AddConstant(MakeJitConstant("IN_B_BLOCK_PITCH", in_b_block_pitch));
        jit.AddConstant(MakeJitConstant("IN_F_BLOCK_PITCH", in_f_block_pitch));
        jit.AddConstant(MakeJitConstant("IN_OFFSET", in_offset));
        return jit;
    }

    KernelsData ConvolutionKernel_mmad_batched::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelsData kd = GetTunedKernelsDataByIndex(params, options);
        if(!kd.empty())
            kd[0].estimatedTime = FORCE_PRIORITY_6;
        return kd;
    }
}