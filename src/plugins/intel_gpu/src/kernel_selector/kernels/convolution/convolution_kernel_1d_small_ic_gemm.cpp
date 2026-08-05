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

// Tile shape of the implicit GEMM.
//
// A lane owns n_per_lane output columns, so TILE_N = simd * n_per_lane while the
// work-group stays one sub-group wide. That is what pays for the inner loop's
// memory traffic: one SLM read of A feeds n_per_lane FMAs instead of one. (A
// lane's columns are simd apart, not adjacent, to keep loads and stores
// coalesced - see the kernel.)
//
// n_per_lane is 2, not 4: TILE_N=64 covers fewer shapes on the B_IN_BOUNDS fast
// path (gemm_n must be a multiple) and halves the accumulator to 32 registers.
constexpr size_t simd = 16;
constexpr size_t n_per_lane = 2;
constexpr size_t tile_m = 16;
constexpr size_t tile_n = simd * n_per_lane;
constexpr size_t tile_k = 32;

// SLM per work-group: the A staging buffer only. B stays in registers, since a
// lane stages the K slice of its own output column and is its only reader. The
// accumulator is fp32 for every dtype (see GetJitConstants), hence the 4 bytes.
constexpr size_t slm_bytes_per_wg = tile_m * tile_k * 4;

// Width of the vector load used to stage B on the unguarded path. Must divide
// tile_k. OpenCL only defines vloadN for N in {2,3,4,8,16}.
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
    // [OC][IC][taps] is already contiguous as [OC][IC * taps], exactly the A
    // matrix this kernel wants, so the default layout avoids a weights reorder.
    return WeightsLayout::oiyx;
}

bool ConvolutionKernel_1d_small_ic_gemm::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const convolution_params&>(p);

    // Shape-agnostic execution is out of scope. Convolution has no dynamic onednn
    // impl either, and the ocl dynamic path is a narrow auto_pad == EXPLICIT
    // fallback (see registry/convolution_impls.cpp), so a dynamic variant would
    // never be selected for a real model while costing the tile-size constant
    // folding this kernel depends on.
    if (params.is_shape_agnostic || params.has_dynamic_tensors()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (params.groups != 1 || params.transposed || params.deformable_mode) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];

    // 1D only, and the long axis has to be the same one everywhere: the kernel
    // indexes input, output and weights through a single LONG_AXIS_IS_X switch, so
    // every short-axis extent must be degenerate, or the axis the host picked from
    // filterSize would disagree with the tensors.
    //
    // padding_end is deliberately not checked: the XY swap updates filterSize,
    // padding_begin, stride and dilation but leaves padding_end alone (see
    // convolution_impl::get_kernel_params), so after a swap it refers to the other
    // axis. This kernel never reads it - the output length comes from the output
    // tensor and the trailing boundary from the in-range test on the staged
    // position.
    const bool long_x = long_axis_is_x(params);
    const size_t short_filter = long_x ? params.filterSize.y : params.filterSize.x;
    const size_t short_in = long_x ? in.Y().v : in.X().v;
    const size_t short_out = long_x ? out.Y().v : out.X().v;
    const size_t short_pad_begin = long_x ? params.padding_begin.y : params.padding_begin.x;

    const bool is_1d = in.Z().v == 1 && out.Z().v == 1 && params.filterSize.z == 1 && short_filter == 1 &&
                       short_in == 1 && short_out == 1 && short_pad_begin == 0;
    if (!is_1d) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (in.Feature().v > max_input_features) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    if (get_taps(params) < min_taps) {
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

    // Read every long-axis value off the same axis so the kernel is correct
    // whether or not the XY swap happened; Validate() has already checked that the
    // short axis is degenerate.
    const bool long_x = long_axis_is_x(params);
    const size_t taps = get_taps(params);
    const size_t in_len = long_x ? in.X().v : in.Y().v;
    const size_t out_len = long_x ? out.X().v : out.Y().v;
    const size_t stride_l = long_x ? params.stride.x : params.stride.y;
    const size_t dilation_l = long_x ? params.dilation.x : params.dilation.y;
    const size_t pad_begin_l = long_x ? params.padding_begin.x : params.padding_begin.y;

    const size_t gemm_m = out.Feature().v;
    const size_t gemm_n = out.Batch().v * out_len;
    const size_t gemm_k = in.Feature().v * taps;

    // Whether the kernel may stage B with unguarded vector loads. That path has no
    // per-element fallback, so each of its assumptions is proved here for the whole
    // iteration space rather than assumed:
    //
    //  - no left padding and the highest position read (last tap of the last
    //    output) still inside the input, so no bounds test is needed;
    //  - a K run is contiguous: dilation 1 and long-axis pitch 1 make consecutive
    //    taps consecutive elements. This holds in practice (DataTensor::SwapXY sets
    //    y.pitch = 1, and X is bfyx's innermost axis otherwise) but is checked;
    //  - tile_k divides taps, so a tile never straddles a k / taps boundary and the
    //    input feature is constant within it;
    //  - tile_n divides gemm_n, so every lane has a valid column;
    //  - b_vec divides tile_k, so the vector loop covers the tile exactly.
    //
    // last_pos is written without subtraction so it cannot wrap.
    const size_t last_pos = (out_len - 1) * stride_l + (taps - 1) * dilation_l;
    const size_t long_pitch = long_x ? in.X().pitch : in.Y().pitch;
    const bool b_in_bounds = pad_begin_l == 0 && dilation_l == 1 && long_pitch == 1 && last_pos < in_len &&
                             taps % tile_k == 0 && gemm_n % tile_n == 0 && tile_k % b_vec == 0;

    jit.AddConstants({
        MakeJitConstant("B_IN_BOUNDS", b_in_bounds),
        MakeJitConstant("B_VEC", b_vec),
        MakeJitConstant("N_PER_LANE", n_per_lane),
        MakeJitConstant("TAPS", taps),
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
        // Which axis the long one landed on, so the kernel's index macros pick
        // the matching argument order. Consistent with get_taps().
        MakeJitConstant("LONG_AXIS_IS_X", long_x),
    });

    // fp32 accumulation for every dtype, including int8: this keeps one code path.
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
    // Must beat convolution_gpu_bfyx_os_iyx_osv16, which is picked for these
    // shapes today and reaches only a few percent of peak on them.
    return FORCE_PRIORITY_2;
}

KernelsData ConvolutionKernel_1d_small_ic_gemm::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

}  // namespace kernel_selector
