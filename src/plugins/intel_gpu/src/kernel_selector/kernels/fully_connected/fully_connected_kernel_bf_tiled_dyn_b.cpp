// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bf_tiled_dyn_b.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <functional>
#include "common_types.h"

static constexpr size_t simd = 16;

namespace kernel_selector {

using namespace fc_kernel_bf_tiled_utils;

FullyConnected_bf_tiled_dyn_b::FullyConnected_bf_tiled_dyn_b()
    : FullyConnectedKernelBase("fully_connected_gpu_bf_tiled_dyn_b") {}

ParamsKey FullyConnected_bf_tiled_dyn_b::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::UINT4);
    k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableWeightsCompression();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey FullyConnected_bf_tiled_dyn_b::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();
    return k;
}

size_t FullyConnected_bf_tiled_dyn_b::SelectTileB(size_t batch_size) {
    // For small batches, tile = batch (exact, no tail)
    if (batch_size <= 8)
        return batch_size;

    // Find largest exact divisor in [8..4] (no tail needed)
    for (size_t t = 8; t >= 4; --t) {
        if (batch_size % t == 0)
            return t;
    }

    // No good exact divisor (primes, 2*prime, etc.):
    // Use TILE_B=8 for maximum throughput; CL entry point handles the tail.
    return 8;
}

bool FullyConnected_bf_tiled_dyn_b::IsBeneficial(const fully_connected_params& params) {
    auto& weights = params.weights;
    auto wt = weights.GetDType();

    // INT4 compressed, F16 input, shape_agnostic only
    if (wt != WeightsType::UINT4 && wt != WeightsType::INT4)
        return false;
    if (!params.compressed)
        return false;
    if (params.inputs[0].GetDType() != Datatype::F16)
        return false;
    if (!params.is_shape_agnostic)
        return false;

    // No SwiGLU support
    if (is_swiglu_fused(params))
        return false;

    // Only beneficial for imbalanced IFM/OFM with sufficient dimension size
    auto ifm = weights.IFM().v;
    auto ofm = weights.OFM().v;
    if (std::min(ifm, ofm) < simd)
        return false;
    if (!(2 * ifm < ofm || ifm > 2 * ofm))
        return false;

    return true;
}

bool FullyConnected_bf_tiled_dyn_b::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];
    auto& output = fc_params.outputs[0];
    auto& weights = fc_params.weights;

    // Only INT4 compressed weights
    auto wt = weights.GetDType();
    if (wt != WeightsType::UINT4 && wt != WeightsType::INT4)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (!fc_params.compressed)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // F16 input only
    if (input.GetDType() != Datatype::F16)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // IFM must be known and even (for INT4 packing)
    if (input.is_dynamic()) {
        auto ifm_size = get_input_bf_size(fc_params).second;
        if (ifm_size == 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    if (weights.IFM().v % 2 != 0)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Block reads: batch pitch must be even for F16
    if (input.Batch().pitch % 2 != 0 && (input.Batch().v > 1 || fc_params.is_shape_agnostic))
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    // For 3D: feature pitch must be even
    if (output.GetLayout() == DataLayout::bfyx && input.Feature().pitch % 2 != 0
        && (input.Feature().v > 1 || fc_params.is_shape_agnostic))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // No padding on spatial dimensions
    if (input.GetLayout() == DataLayout::bfyx) {
        if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // No 4D output
    if (output.GetLayout() == DataLayout::bfyx) {
        if (input.X().v > 1)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Reject dynamic quantization targets (let bf_tiled SLM handle those)
    if (is_weight_dyn_quantizable(fc_params) && should_dynamic_quantize(fc_params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // This kernel is designed for dynamic batch dispatch only.
    // For static shapes, bf_tiled already selects optimal TILE_B at compile time.
    if (!fc_params.is_shape_agnostic)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // SwiGLU fused op is not supported (requires OUTER_OFM=2 / SWIGLU_LENGTH handling)
    if (is_swiglu_fused(fc_params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    // Only beneficial when IFM and OFM are highly imbalanced with sufficient size.
    // For near-square shapes (IFM ≈ OFM), bf_tiled's dispatch is already efficient.
    // Accept when: min(IFM,OFM) >= SIMD  and  (2*IFM < OFM  or  IFM > 2*OFM)
    {
        auto ifm = weights.IFM().v;
        auto ofm = weights.OFM().v;
        if (std::min(ifm, ofm) < simd)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        if (!(2 * ifm < ofm || ifm > 2 * ofm))
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}


FullyConnected_bf_tiled_dyn_b::tune_params
FullyConnected_bf_tiled_dyn_b::GetTuneParams(const fully_connected_params& params) const {
    // Same base config as static_b16 (optimized for INT4 on iGPU)
    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16)
        return tune_params(1, 1, 4, 1, 1, EXE_MODE_DEFAULT);
    else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2)
        return tune_params(2, 1, 2, 1, 1, EXE_MODE_DEFAULT);
    else  // os_is_yx_osv32_isv2 (default)
        return tune_params(2, 1, 4, 1, 1, EXE_MODE_DEFAULT);
}

FullyConnected_bf_tiled_dyn_b::DispatchData
FullyConnected_bf_tiled_dyn_b::SetDefault(const fully_connected_params& params, int autoTuneIndex, int kernel_number) const {
    auto dispatchData = Parent::SetDefault(params);
    auto tparams = GetTuneParams(params);

    // For initial compilation use TILE_B=8 as default (will be updated at runtime)
    constexpr size_t default_tile_b = 8;
    auto bf_size = get_output_aligned_bf_size(params, false);
    size_t batch = bf_size.first;
    size_t tile_b = (batch > 0) ? SelectTileB(batch) : default_tile_b;

    auto threads = get_output_aligned_bf_size(params, true,
                                              static_cast<uint32_t>(tile_b),
                                              static_cast<int32_t>(tparams.tile_ofm * simd));
    auto batch_threads = threads.first;
    auto feature_threads = threads.second;

    dispatchData.gws[0] = feature_threads * batch_threads * simd;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = simd;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.tile_m = static_cast<uint32_t>(tile_b);
    dispatchData.tile_n = tparams.tile_ofm;
    dispatchData.tile_mk = tparams.tile_ifm;
    dispatchData.tile_nk = tparams.tile_k;
    dispatchData.outer_n = 1;
    dispatchData.tile_ms = tparams.dispatch_bsv;
    dispatchData.tile_ns = tparams.dispatch_fsv;
    dispatchData.use_slm = false;

    return dispatchData;
}

void FullyConnected_bf_tiled_dyn_b::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const fully_connected_params&>(params);

        auto dispatchData = SetDefault(prim_params);
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsPriority FullyConnected_bf_tiled_dyn_b::GetKernelsPriority(const Params& params) const {
    // Low priority: dyn_b is integrated into bf_tiled as a runtime sub-kernel.
    // Standalone is only used when explicitly forced via force_implementations (tests).
    return FORCE_PRIORITY_9;
}

JitConstants FullyConnected_bf_tiled_dyn_b::GetJitConstants(const fully_connected_params& params,
                                                             const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    size_t tile_k_ofm = dispatchData.tile_nk * dispatchData.tile_n;
    size_t tile_k_ofm_packed = tile_k_ofm;

    bool add_decompress_scale_post_op = false;
    WeightsType weights_dt = params.weights.GetDType();
    if (weights_dt == WeightsType::UINT4 || weights_dt == WeightsType::INT4) {
        tile_k_ofm_packed /= 2;
        jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE", weights_dt, tile_k_ofm));
        const size_t scale_group_size = get_scale_group_size(params);
        if (scale_group_size % simd == 0)
            add_decompress_scale_post_op = true;
    }

    // W_IDX and TILE_OFM_PER_OSV_SIZE based on weight layout
    if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        jit.AddConstant(MakeJitConstant("W_IDX", "fi * TILE_K + kii"));
    } else {
        jit.AddConstant(MakeJitConstant("W_IDX", "kii * TILE_OFM + fi"));
    }

    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16 && dispatchData.tile_n == 2) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 0.5f));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2 && dispatchData.tile_n == 1) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 2));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2 && dispatchData.tile_n == 2) {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 2));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OFM_PER_OSV_SIZE", 1));
    }

    if (add_decompress_scale_post_op)
        jit.AddConstant(MakeJitConstant("DECOMPRESSION_SCALE_POST_OP", 1));

    jit.AddConstant(MakeJitConstant("DYNAMIC_QUANTIZE", 0));
    jit.AddConstant(MakeJitConstant("IFM_SIZE", get_input_bf_size(params).second));
    jit.AddConstant(MakeJitConstant("SIMD", simd));

    // TILE_B is not used directly in the kernel dispatch,
    // but needed for CONST_LOOP unroll generation (max tile = 8).
    jit.AddConstant(MakeJitConstant("TILE_OFM", dispatchData.tile_n));
    jit.AddConstant(MakeJitConstant("TILE_IFM", dispatchData.tile_mk));
    jit.AddConstant(MakeJitConstant("TILE_K", dispatchData.tile_nk));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM", tile_k_ofm));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM_PACKED", tile_k_ofm_packed));
    jit.AddConstant(MakeJitConstant("DISPATCH_BSV", dispatchData.tile_ms));
    jit.AddConstant(MakeJitConstant("DISPATCH_FSV", dispatchData.tile_ns));

    // Generate CONST_LOOP macros up to max FORCED_TILE_B = 8
    jit.Merge(MakeConstantLoopUnrollJitConstants(8));

    bool realign_fp16_offset = params.inputs[0].GetDType() == Datatype::F16 && params.inputs[0].GetFirstElementOffset() % 2 != 0;
    jit.AddConstant(MakeJitConstant("REALIGN_FP16_OFFSET", realign_fp16_offset));

    auto activation_dt = GetActivationType(params);
    auto accumulator_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));

    // Output layout constants
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        auto tile_in_b_pitch = (params.inputs[0].Feature().pitch == 0)
                                   ? get_input_bf_size(params).second
                                   : params.inputs[0].Feature().pitch;
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", tile_in_b_pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        auto tile_in_b_pitch = (params.inputs[0].Batch().pitch == 0)
                                   ? get_input_bf_size(params).second
                                   : params.inputs[0].Batch().pitch;
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", tile_in_b_pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    // Fused ops setup
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order_scalar = { "(out_b + bi)", "(out_f + sglid)", "0", "0" };
        std::vector<std::string> idx_order_vec = { "(out_b + bi)", "(out_f + sglid + fi * SIMD)", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
            idx_order_scalar = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "(out_f + sglid)", "0" };
            idx_order_vec = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "(out_f + sglid + fi * SIMD)", "0" };
        }

        FusedOpsConfiguration conf_scalar = { "_SCALAR", idx_order_scalar, "activated[bi]", activation_dt, 1 };
        FusedOpsConfiguration conf_vec = { "_VEC", idx_order_vec, "activated[bi][fi]", activation_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_scalar, conf_vec }));
    }

    return jit;
}

KernelsData FullyConnected_bf_tiled_dyn_b::GetKernelsData(const Params& params) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto tparams = GetTuneParams(fc_params);

    // Determine optimal weight layout
    WeightsLayout weights_layout = WeightsLayout::os_iyx_osv16;
    auto output_f = get_output_aligned_bf_size(fc_params, false).second;
    if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16
        && (fc_params.weights.GetDType() == WeightsType::INT4 || fc_params.weights.GetDType() == WeightsType::UINT4)) {
        if ((fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2)
            && is_weight_horizontal(fc_params, output_f)) {
            weights_layout = WeightsLayout::os_is_yx_osv64_isv2;
        } else if ((fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_iyx_osv16)
                   && is_weight_vertical(fc_params, output_f)) {
            weights_layout = WeightsLayout::os_iyx_osv16;
        } else if (fc_params.weights.GetLayout() == WeightsLayout::oiyx || fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
            weights_layout = WeightsLayout::os_is_yx_osv32_isv2;
        }
    }

    auto kernels_data = GetCommonKernelsData(params,
                                             fc_params.inputs[0].GetLayout(),
                                             weights_layout,
                                             tparams.exec_options,
                                             -1,
                                             0);

    if (!kernels_data.empty()) {
        GetUpdateDispatchDataFunc(kernels_data[0]);
    }

    return kernels_data;
}

Datatype FullyConnected_bf_tiled_dyn_b::GetAccumulatorType(const fully_connected_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    auto wei_dt = params.weights.GetDType();

    // F16 input + INT4 weights: force F32 accumulator to prevent overflow.
    // F16 max is 65504; accumulating 128 products of F16*INT4 (scale group size)
    // can exceed this, overflowing to inf and cascading to NaN.
    if (in_dt == Datatype::F16 && (wei_dt == WeightsType::INT4 || wei_dt == WeightsType::UINT4))
        return Datatype::F32;

    return Parent::GetAccumulatorType(params);
}

}  // namespace kernel_selector
