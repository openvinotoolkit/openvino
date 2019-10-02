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

#include "convolution_kernel_mmad_32x32sg_slm_int8.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static const size_t _SG_TILE_M = 32;
static const size_t _SG_TILE_N = 32;
static const size_t _SG_SIZE = 8;         // sub group size
static const size_t _TILES_PER_SG_X = 1;  // Persistent threads
static const size_t _TILES_PER_SG_Y = 1;  // Persistent threads

ParamsKey ConvolutionKernel_mmad_32x32sg_slm_int8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableOutputLayout(DataLayout::fs_bs_yx_bsv4_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBatching();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    k.DisableTuning();
    return k;
}

bool ConvolutionKernel_mmad_32x32sg_slm_int8::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    // make sure it's 1x1 conv
    if (cp.filterSize.x != 1 || cp.filterSize.y != 1)
        return false;

    // make sure stride is 1x1
    if (cp.stride.x != 1 || cp.stride.y != 1)
        return false;

    // input padding not supported
    if (cp.inputs[0].X().pad.Total() != 0 || cp.inputs[0].Y().pad.Total() != 0 ||
        cp.inputs[0].Feature().pad.Total() != 0 || cp.inputs[0].Batch().pad.Total() != 0)
        return false;

    // input and output spatial sizes must match
    if (!(cp.output.X().v == cp.inputs[0].X().v) || !(cp.output.Y().v == cp.inputs[0].Y().v))
        return false;

    const auto m = cp.output.X().v * cp.output.Y().v * cp.output.Batch().v;
    const auto k = cp.inputs[0].Feature().v;
    const auto n = cp.output.Feature().v;

    if (m % 32 != 0)  // Matrix size M, Must be mutliple of 32
        return false;

    if (k % 32 != 0)  // Matrix size K, Must be multiple of 32
        return false;

    if (n % 32 != 0)  // Matrix size N, Must be mutliple of 32
        return false;

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_mmad_32x32sg_slm_int8::SetDefault(const convolution_params& arg,
                                                                                        int) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(arg);

    runInfo.effiency = FORCE_PRIORITY_2;

    size_t mat_m = arg.output.X().v * arg.output.Y().v * arg.output.Batch().v;
    size_t mat_n = arg.output.Feature().v;

    size_t _MATRIX_M = mat_m;
    size_t _MATRIX_N = mat_n;

    size_t _WG_TILE_M = 32;
    size_t _WG_TILE_N = 32;

    // Calculate number of threads needed
    const size_t threadsX = (_MATRIX_N / (_SG_TILE_N / _SG_SIZE)) / _TILES_PER_SG_X;
    const size_t threadsY = (_MATRIX_M / _SG_TILE_M) / _TILES_PER_SG_Y;

    // Define execution setup for kernel:
    size_t globalWorkSize[3] = {threadsX, threadsY, 1};
    size_t localWorkSize[3] = {_SG_SIZE * _WG_TILE_N / _SG_TILE_N, _WG_TILE_M / _SG_TILE_M, 1};

    runInfo.gws0 = globalWorkSize[0];
    runInfo.gws1 = globalWorkSize[1];
    runInfo.gws2 = globalWorkSize[2];

    runInfo.lws0 = localWorkSize[0];
    runInfo.lws1 = localWorkSize[1];
    runInfo.lws2 = localWorkSize[2];

    return runInfo;
}

JitConstants ConvolutionKernel_mmad_32x32sg_slm_int8::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("WG_TILE_M", 32));  // Work-Group tile size M, Must be mutliple of 32
    jit.AddConstant(MakeJitConstant("WG_TILE_N", 32));  // Work-Group tile size N, Must be mutliple of 32
    jit.AddConstant(MakeJitConstant("TILES_PER_SG_X", _TILES_PER_SG_X));
    jit.AddConstant(MakeJitConstant("TILES_PER_SG_Y", _TILES_PER_SG_Y));

    // Do not change values below
    jit.AddConstant(MakeJitConstant("DIM_X", 0));
    jit.AddConstant(MakeJitConstant("DIM_Y", 1));
    jit.AddConstant(MakeJitConstant("MATRIX_SMALL_K", 32));
    jit.AddConstant(MakeJitConstant("MATRIX_SMALL_K_BFLOAT", 16));
    jit.AddConstant(MakeJitConstant("SG_TILE_M", _SG_TILE_M));
    jit.AddConstant(MakeJitConstant("SG_TILE_N", _SG_TILE_N));
    jit.AddConstant(MakeJitConstant("SG_SIZE", _SG_SIZE));
    jit.AddConstant(MakeJitConstant("SIMD_LANE_M", "SG_TILE_M"));
    jit.AddConstant(MakeJitConstant("SIMD_LANE_N", "(SG_TILE_N / SG_SIZE)"));
    jit.AddConstant(MakeJitConstant("WG_SIZE", "(SG_SIZE * WG_TILE_N / SG_TILE_N) * (WG_TILE_M / SG_TILE_M)"));

    jit.AddConstant(MakeJitConstant("COMPILE_KERNELS", ""));
    jit.AddConstant(MakeJitConstant("TILED_GLOBAL_LAYOUT", ""));
    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_GLOBAL_LAYOUT", ""));

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    auto m = output.X().v * output.Y().v * output.Batch().v;
    auto k = input.Feature().v;
    auto n = output.Feature().v;

    jit.AddConstant(MakeJitConstant("MATRIX_M", m));  // Matrix size M, Must be mutliple of 32 and multiple of WG_TILE_M
    jit.AddConstant(MakeJitConstant("MATRIX_K", k));  // Matrix size K, Must be mutliple of 32
    jit.AddConstant(MakeJitConstant("MATRIX_N", n));  // Matrix size N, Must be mutliple of 32 and multiple of WG_TILE_N

    const size_t out_x_pitch = 32 * 4;
    const size_t out_y_pitch = 32 * 4 * params.output.X().LogicalDimPadded();
    const size_t out_b_block_pitch = out_y_pitch * params.output.Y().LogicalDimPadded();
    const size_t out_f_block_pitch = out_b_block_pitch * ((params.output.Batch().v + 3) / 4);
    const size_t out_offset = out_x_pitch * params.output.X().pad.before + out_y_pitch * params.output.Y().pad.before;

    jit.AddConstant(MakeJitConstant("OUT_X_PITCH", out_x_pitch));
    jit.AddConstant(MakeJitConstant("OUT_Y_PITCH", out_y_pitch));
    jit.AddConstant(MakeJitConstant("OUT_B_BLOCK_PITCH", out_b_block_pitch));
    jit.AddConstant(MakeJitConstant("OUT_F_BLOCK_PITCH", out_f_block_pitch));
    jit.AddConstant(MakeJitConstant("OUT_OFFSET", out_offset));

    return jit;
}

KernelsData ConvolutionKernel_mmad_32x32sg_slm_int8::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    KernelsData kd = GetCommonKernelsData(params, options);
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_2;  // _3
    return kd;
}
}  // namespace kernel_selector