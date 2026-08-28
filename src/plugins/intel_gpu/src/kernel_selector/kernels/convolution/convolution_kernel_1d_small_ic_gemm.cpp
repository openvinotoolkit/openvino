// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_1d_small_ic_gemm.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <algorithm>
#include <string>
#include <vector>

namespace kernel_selector {
namespace {

// Tile shape of the implicit GEMM. A lane owns n_per_lane output columns, so one
// SLM read of A feeds n_per_lane FMAs. n_per_lane is 2, not 4: TILE_N=64 covers
// fewer shapes on the B_IN_BOUNDS path and doubles the accumulator.
constexpr size_t simd = 16;
constexpr size_t n_per_lane = 2;
constexpr size_t tile_m = 16;
constexpr size_t tile_n = simd * n_per_lane;
constexpr size_t tile_k = 32;

// SLM per work-group: the A staging buffer only, B stays in registers. 4 bytes
// because the accumulator is fp32 for every dtype (see GetJitConstants).
constexpr size_t slm_bytes_per_wg = tile_m * tile_k * 4;

// Width of the vector load used to stage B. OpenCL only defines vloadN for N in
// {2,3,4,8,16}.
constexpr size_t b_vec = 8;
static_assert(tile_k % b_vec == 0, "B_VEC must divide TILE_K");

}  // namespace

ParamsKey ConvolutionKernel_1d_small_ic_gemm::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();

    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);

    // Deliberately no EnableDynamicShapesSupport(): see Validate().
    return k;
}

DeviceFeaturesKey ConvolutionKernel_1d_small_ic_gemm::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

WeightsLayout ConvolutionKernel_1d_small_ic_gemm::GetPreferredWeightsLayout(const convolution_params&) const {
    // Already contiguous as [OC][IC * filter length], i.e. the A matrix, so no
    // weights reorder is needed.
    return WeightsLayout::oiyx;
}

bool ConvolutionKernel_1d_small_ic_gemm::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const convolution_params&>(p);

    // Static shapes only: the shape is folded into the jit constants below.
    if (params.is_shape_agnostic || params.has_dynamic_tensors()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (params.groups != 1 || params.transposed || params.deformable_mode) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];

    if (in.Feature().v > max_input_features) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (get_filter_len(params) < min_filter_len) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // 1D only: the kernel indexes every tensor through a single LONG_AXIS_IS_X
    // switch, so all axes but the long one must be degenerate and unpadded.
    // padding_end is not checked - the XY swap leaves it referring to the other axis
    // (see convolution_impl::get_kernel_params) and this kernel never reads it.
    const bool long_x = long_axis_is_x(params);
    const size_t short_filter = long_x ? params.filterSize.y : params.filterSize.x;
    const size_t short_in = long_x ? in.Y().v : in.X().v;
    const size_t short_out = long_x ? out.Y().v : out.X().v;
    const size_t short_pad_begin = long_x ? params.padding_begin.y : params.padding_begin.x;

    const bool is_1d = short_filter == 1 && short_in == 1 && short_out == 1 && short_pad_begin == 0 &&
                       params.filterSize.z == 1 && in.Z().v == 1 && out.Z().v == 1;
    if (!is_1d) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    // The staging buffers must fit in the device's SLM budget.
    if (slm_bytes_per_wg > params.engineInfo.maxLocalMemSize) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (!IsSIMDSizeSupported(params.engineInfo, simd)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

JitConstants ConvolutionKernel_1d_small_ic_gemm::GetJitConstants(const convolution_params& params,
                                                                const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];

    // Read every value off the long axis, so the XY swap does not matter.
    const bool long_x = long_axis_is_x(params);
    const size_t filter_len = get_filter_len(params);
    const size_t in_len = long_x ? in.X().v : in.Y().v;
    const size_t out_len = long_x ? out.X().v : out.Y().v;
    const size_t stride_l = long_x ? params.stride.x : params.stride.y;
    const size_t dilation_l = long_x ? params.dilation.x : params.dilation.y;
    const size_t pad_begin_l = long_x ? params.padding_begin.x : params.padding_begin.y;

    const size_t gemm_m = out.Feature().v;
    const size_t gemm_n = out.Batch().v * out_len;
    const size_t gemm_k = in.Feature().v * filter_len;

    // The unguarded vector loads for B have no per-element fallback, so every
    // position must be provably in range (no left padding, last read inside the
    // input) and a K run contiguous (dilation 1, unit pitch), with tile_k and tile_n
    // dividing the filter length and gemm_n so no tile is ragged. last_pos avoids
    // subtraction so it cannot wrap.
    const size_t last_pos = (out_len - 1) * stride_l + (filter_len - 1) * dilation_l;
    const size_t long_pitch = long_x ? in.X().pitch : in.Y().pitch;
    const bool b_in_bounds = pad_begin_l == 0 && dilation_l == 1 && long_pitch == 1 && last_pos < in_len &&
                             filter_len % tile_k == 0 && gemm_n % tile_n == 0 && tile_k % b_vec == 0;

    jit.AddConstants({
        MakeJitConstant("B_IN_BOUNDS", b_in_bounds),
        MakeJitConstant("B_VEC", b_vec),
        MakeJitConstant("N_PER_LANE", n_per_lane),
        MakeJitConstant("FILTER_LEN", filter_len),
        MakeJitConstant("IN_LEN", in_len),
        MakeJitConstant("OUT_LEN", out_len),
        MakeJitConstant("STRIDE_L", stride_l),
        MakeJitConstant("DILATION_L", dilation_l),
        MakeJitConstant("PAD_BEGIN_L", pad_begin_l),
        MakeJitConstant("GEMM_M", gemm_m),
        MakeJitConstant("GEMM_N", gemm_n),
        MakeJitConstant("GEMM_K", gemm_k),
        MakeJitConstant("TILE_M", tile_m),
        MakeJitConstant("TILE_N", tile_n),
        MakeJitConstant("TILE_K", tile_k),
        MakeJitConstant("SIMD", simd),
        MakeJitConstant("LONG_AXIS_IS_X", long_x),
    });

    // fp32 accumulation for every dtype, including int8, to keep one code path.
    const Datatype accumulator_dt = Datatype::F32;
    const Datatype activation_dt = Datatype::F32;

    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    return jit;
}

ConvolutionKernel_1d_small_ic_gemm::Parent::DispatchData ConvolutionKernel_1d_small_ic_gemm::SetDefault(
    const convolution_params& params,
    int autoTuneIndex) const {
    DispatchData dispatchData = Parent::SetDefault(params, autoTuneIndex);

    const auto& out = params.outputs[0];
    const size_t out_len = long_axis_is_x(params) ? out.X().v : out.Y().v;

    const size_t gemm_m = out.Feature().v;
    const size_t gemm_n = out.Batch().v * out_len;

    dispatchData.gws = {CeilDiv(gemm_n, tile_n) * simd, CeilDiv(gemm_m, tile_m), 1};
    dispatchData.lws = {simd, 1, 1};

    return dispatchData;
}

KernelsPriority ConvolutionKernel_1d_small_ic_gemm::GetKernelsPriority(const Params& /*params*/) const {
    // Must beat convolution_gpu_bfyx_os_iyx_osv16, which is picked for these shapes
    // today and reaches only a few percent of peak.
    return FORCE_PRIORITY_2;
}

KernelsData ConvolutionKernel_1d_small_ic_gemm::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

}  // namespace kernel_selector
