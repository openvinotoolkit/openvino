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


#include "pooling_kernel_gpu_fs_bs_yx_bsv4_fsv32_simd32.h"

namespace kernel_selector {
ParamsKey PoolingKerneGPU_fs_bs_yx_bsv4_fsv32_simd32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnablePoolType(PoolType::MAX);
    //        k.EnablePoolType(PoolType::AVG);
    k.EnablePoolRemainder(PoolRemainder::FLOOR);
    k.EnablePoolRemainder(PoolRemainder::CEIL);
    k.EnablePoolKernelDividerMode(KernelDividerMode::FIXED);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC);
    k.EnablePoolKernelDividerMode(KernelDividerMode::DYNAMIC_WITH_PADDING);
    k.EnableDifferentTypes();
    return k;
}

size_t static get_batch_sub_groups_count(const pooling_params& params) {
    if (params.inputs[0].Batch().v % 32 == 0)
        return 8;  // divided by 4 because we process 4 batches per subgroup
    return 1;
}

PoolingKernelBase::DispatchData PoolingKerneGPU_fs_bs_yx_bsv4_fsv32_simd32::SetDefault(
    const pooling_params& params) const {
    constexpr int simdSize = 32;

    DispatchData runInfo = PoolingKernelBase::SetDefault(params);

    runInfo.gws0 = params.output.X().v;
    runInfo.gws1 = params.output.Y().v;
    // we got fs_bs_yx_bsv4_fsv32 format, we process 4 batches and 4 features per workitem
    runInfo.gws2 = (RoundUp(params.output.Feature().v, 32) * RoundUp(params.output.Batch().v, 4)) / (4);  // *4);

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = simdSize * get_batch_sub_groups_count(params);

    return runInfo;
}

JitConstants PoolingKerneGPU_fs_bs_yx_bsv4_fsv32_simd32::GetJitConstants(const pooling_params& params,
                                                                         DispatchData kd) const {
    auto jit = PoolingKernelBase::GetJitConstants(params, kd);

    const size_t in_x_pitch = 32 * 4;
    const size_t in_y_pitch = 32 * 4 * params.inputs[0].X().LogicalDimPadded();
    const size_t in_b_block_pitch = in_y_pitch * params.inputs[0].Y().LogicalDimPadded();
    const size_t in_f_block_pitch = in_b_block_pitch * ((params.inputs[0].Batch().v + 3) / 4);
    const size_t in_offset =
        in_x_pitch * params.inputs[0].X().pad.before + in_y_pitch * params.inputs[0].Y().pad.before;

    jit.AddConstant(MakeJitConstant("IN_X_PITCH", in_x_pitch));
    jit.AddConstant(MakeJitConstant("IN_Y_PITCH", in_y_pitch));
    jit.AddConstant(MakeJitConstant("IN_B_BLOCK_PITCH", in_b_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_F_BLOCK_PITCH", in_f_block_pitch));
    jit.AddConstant(MakeJitConstant("IN_OFFSET", in_offset));
    jit.AddConstant(MakeJitConstant("BATCH_SG_COUNT", get_batch_sub_groups_count(params)));

    return jit;
}

KernelsData PoolingKerneGPU_fs_bs_yx_bsv4_fsv32_simd32::GetKernelsData(const Params& params,
                                                                       const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_1);
}
}  // namespace kernel_selector