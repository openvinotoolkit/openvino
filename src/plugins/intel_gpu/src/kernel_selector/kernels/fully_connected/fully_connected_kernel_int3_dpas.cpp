// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_int3_dpas.h"
#include "fully_connected_kernel_bf_tiled.h"
#include "kernel_selector_utils.h"
#include "common_types.h"

#include <algorithm>
#include <vector>

namespace kernel_selector {

namespace {
constexpr size_t simd = 16;
constexpr size_t k_chunk = 32;  // u3 values per granule, and the DPAS K step
constexpr size_t osv = 16;      // output channels per weights block
constexpr size_t min_quantize_group_size = simd * 2;
// Batch at or above which the matrix-engine variant wins over the K-split one.
constexpr size_t dpas_min_batch = 8;

using gemm_config = FullyConnected_int3_dpas::gemm_config;
using fc_kernel_bf_tiled_utils::get_input_bf_size;
using fc_kernel_bf_tiled_utils::get_output_aligned_bf_size;

// Quantization groups along K carried by the decompression scale, which holds one
// row of groups per output channel.
size_t get_scale_groups_k(const fully_connected_params& params) {
    const size_t rows = params.weights.OFM().v;
    const size_t total = params.decompression_scale.LogicalSize();
    if (rows == 0 || total == 0 || (total % rows) != 0)
        return 0;
    return total / rows;
}

size_t get_wei_scale_group_size(const fully_connected_params& params) {
    const size_t groups = get_scale_groups_k(params);
    const size_t ifm = params.weights.IFM().v;
    if (groups == 0 || ifm == 0 || (ifm % groups) != 0)
        return 0;
    return ifm / groups;
}

// Row stride of the activation tensor, in elements.
size_t get_input_b_pitch(const fully_connected_params& params) {
    const auto pitch = (params.outputs[0].GetLayout() == DataLayout::bfyx) ? params.inputs[0].Feature().pitch
                                                                          : params.inputs[0].Batch().pitch;
    return (pitch == 0) ? get_input_bf_size(params).second : static_cast<size_t>(pitch);
}

// The decompression scale and zero point are read as whole elements, so a packed
// type would silently index the wrong value.
bool is_addressable_dtype(Datatype dt) {
    return cldnn::one_of(dt, {Datatype::F16, Datatype::F32, Datatype::INT8, Datatype::UINT8, Datatype::INT32, Datatype::UINT32});
}

// Deliberately not bf_tiled's get_dynamic_quantize_group_size: its per-token branch
// returns the weight scale group size, which can be the whole of IFM. The group is
// the unit the DPAS path decodes into registers at once (CHUNKS_PER_GROUP granules,
// each an int8) and stages through SLM, so it has to stay small and bounded.
size_t get_quantize_group_size(const fully_connected_params& params) {
    if (!params.compressed || params.decompression_scale.Feature().v == 0)
        return 0;

    const size_t ifm = get_input_bf_size(params).second;
    if (ifm == 0)
        return 0;

    const size_t scale_group_size = get_wei_scale_group_size(params);
    if (scale_group_size == 0)
        return 0;

    // A group is also the unit at which the integer accumulator is drained and
    // rescaled, so the weight scale - and the zero point, which is folded in via
    // the activation sum - must be constant across it.
    size_t zp_group_size = 0;
    if (params.has_decompression_zp && !params.scalar_zp) {
        const size_t zp_groups = params.decompression_zero_point.Feature().v;
        if (zp_groups == 0)
            return 0;
        zp_group_size = params.weights.IFM().v / zp_groups;
    }

    for (size_t candidate : {size_t{128}, size_t{64}, size_t{32}}) {
        if (candidate < min_quantize_group_size)
            continue;
        if (params.dynamic_quantization_group_size < candidate)
            continue;
        if ((ifm % candidate) != 0 || (scale_group_size % candidate) != 0)
            continue;
        if (zp_group_size != 0 && (zp_group_size % candidate) != 0)
            continue;
        return candidate;
    }

    return 0;
}

size_t get_quantized_input_size(const fully_connected_params& params) {
    const auto bf = get_input_bf_size(params);
    return std::max(params.inputs[0].PhysicalSize(), bf.first * bf.second);
}

// The sg_m subgroups split one staging iteration's granules between them, so the
// iteration has to cover whole quantization groups and split evenly.
bool is_valid_sg_m(const fully_connected_params& params, size_t sg_m) {
    const size_t group_size = get_quantize_group_size(params);
    if (group_size < k_chunk)
        return sg_m == 1;
    const size_t chunks_per_group = group_size / k_chunk;
    const size_t groups_k = get_input_bf_size(params).second / group_size;
    const size_t groups_per_iter = (sg_m > chunks_per_group) ? sg_m / chunks_per_group : 1;
    const size_t chunks_per_iter = groups_per_iter * chunks_per_group;
    return (chunks_per_iter % sg_m) == 0 && (groups_k % groups_per_iter) == 0;
}

// Subgroups sharing one weight decode through SLM, by row count, with 32-row tiles. Device-timed on the Qwen3-8B shapes (N and K
// from 4096 to 12288): each doubling of sg_m pays off once the workgroup's
// 32 * sg_m rows are mostly filled, up to ~21 TOPS at sg_m 8 versus ~10 at 1.
size_t get_dense_sg_m(size_t rows) {
    if (rows >= 384)
        return 8;
    if (rows >= 96)
        return 4;
    if (rows >= 48)
        return 2;
    return 1;
}

// Column blocks per subgroup and rows per subgroup of the v2 path. Its 16 x 64
// tiles need 256 registers per thread (see add_large_grf_option).
constexpr size_t v2_nb = 4;
constexpr size_t v2_tile_m = 16;
// Row counts from which v2 takes over with 8 and with 16 subgroups per workgroup.
// A workgroup covers 16 * sg_m rows, so each step pays off once it is mostly full.
// Below 96 rows the 32-row tiles of the original path are as fast or faster.
constexpr size_t v2_min_rows_sg8 = 96;
constexpr size_t v2_min_rows_sg16 = 192;

// The v2 path relies on 2D block reads and split work-group barriers (Xe2 and
// later), shares one decode across nb column blocks and folds a scalar zero
// point into a per-row term, so it is limited to FCs whose zero point is
// scalar or absent and whose output channels fill whole nb * 16 column tiles.
bool supports_v2(const fully_connected_params& params) {
    if (params.engineInfo.arch < gpu_arch::xe2)
        return false;
    if (params.has_decompression_zp && !params.scalar_zp)
        return false;
    const size_t group_size = get_quantize_group_size(params);
    if (group_size < 2 * k_chunk || ((group_size / k_chunk) % 2) != 0)
        return false;
    const size_t ofm = get_output_aligned_bf_size(params, false).second;
    if ((ofm % (v2_nb * osv)) != 0)
        return false;
    // 2D block reads need a surface width and pitch of at least 64 bytes.
    return get_input_bf_size(params).second >= 64;
}

bool is_valid_v2_sg_m(const fully_connected_params& params, size_t sg_m) {
    const size_t group_size = get_quantize_group_size(params);
    if (sg_m < 2 || group_size < k_chunk)
        return false;
    const size_t gran_per_group = (group_size / k_chunk) * v2_nb;
    const size_t groups_k = get_input_bf_size(params).second / group_size;
    const size_t groups_per_iter = (sg_m > 2 * gran_per_group) ? sg_m / gran_per_group : 2;
    return ((groups_per_iter * gran_per_group) % sg_m) == 0 && (groups_per_iter % 2) == 0 &&
           (groups_k % groups_per_iter) == 0;
}

void add_large_grf_option(clKernelData& kernel) {
    auto& options = kernel.code.kernelString->options;
    const std::string small_grf = " -ze-exp-register-file-size 128";
    const auto pos = options.find(small_grf);
    if (pos != std::string::npos)
        options.erase(pos, small_grf.size());
    options += " -cl-intel-256-GRF-per-thread";
}

gemm_config get_v2_config(size_t sg_m, size_t min_rows) {
    gemm_config cfg;
    cfg.dpas = true;
    cfg.v2 = true;
    cfg.tile_m = v2_tile_m;
    cfg.sg_m = sg_m;
    cfg.nb = v2_nb;
    cfg.min_rows = min_rows;
    return cfg;
}

// A shape-agnostic dense FC compiles one DPAS kernel per entry and picks one per
// inference by row count, since a single compiled config cannot serve both a
// 30-row and a 2000-row prompt well. Every list has four entries, which is how
// the update function recognises the dense variant set from the kernel count.
constexpr size_t dense_variant_count = 4;

std::vector<gemm_config> get_dense_variants(const fully_connected_params& params) {
    std::vector<gemm_config> configs;
    const size_t sg_ms[] = {1, 2};
    const size_t min_rows[] = {0, 48};
    for (size_t i = 0; i < 2; ++i) {
        gemm_config cfg;
        cfg.dpas = true;
        cfg.tile_m = 32;
        cfg.sg_m = sg_ms[i];
        cfg.min_rows = min_rows[i];
        configs.push_back(cfg);
    }
    if (supports_v2(params) && is_valid_v2_sg_m(params, 8) && is_valid_v2_sg_m(params, 16)) {
        configs.push_back(get_v2_config(8, v2_min_rows_sg8));
        configs.push_back(get_v2_config(16, v2_min_rows_sg16));
    } else {
        for (size_t sg_m : {size_t{4}, size_t{8}}) {
            gemm_config cfg;
            cfg.dpas = true;
            cfg.tile_m = 32;
            cfg.sg_m = sg_m;
            cfg.min_rows = (sg_m == 4) ? 96 : 384;
            configs.push_back(cfg);
        }
    }
    return configs;
}

bool use_dense_variants(const fully_connected_params& params) {
    if (!params.is_shape_agnostic || get_quantize_group_size(params) == 0)
        return false;
    for (const auto& cfg : get_dense_variants(params)) {
        if (!cfg.v2 && !is_valid_sg_m(params, cfg.sg_m))
            return false;
    }
    return true;
}

gemm_config get_dpas_config(const fully_connected_params& params) {
    gemm_config cfg;
    cfg.dpas = true;
    cfg.tile_m = 32;
    cfg.sg_m = 1;

    const size_t group_size = get_quantize_group_size(params);
    if (group_size == 0)
        return cfg;

    // A shape-agnostic kernel is compiled before the row count is known. It gets
    // 32 x 1 when get_dense_variants does not apply.
    if (params.is_shape_agnostic)
        return cfg;

    const size_t rows = get_input_bf_size(params).first;
    if (rows <= 8) {
        cfg.tile_m = 8;
    } else if (rows <= 16) {
        cfg.tile_m = 16;
    } else {
        if (rows >= v2_min_rows_sg8 && supports_v2(params)) {
            const size_t sg_m = (rows >= v2_min_rows_sg16) ? 16 : 8;
            if (is_valid_v2_sg_m(params, sg_m))
                return get_v2_config(sg_m, 0);
        }
        for (size_t sg_m = get_dense_sg_m(rows); sg_m > 1; sg_m /= 2) {
            if (is_valid_sg_m(params, sg_m)) {
                cfg.sg_m = sg_m;
                break;
            }
        }
    }

    return cfg;
}

gemm_config get_scalar_config(const fully_connected_params& params) {
    gemm_config cfg;
    cfg.dpas = false;
    cfg.tile_m = 1;
    cfg.sg_k = 1;

    const size_t group_size = get_quantize_group_size(params);
    if (group_size == 0)
        return cfg;

    const size_t groups_k = get_input_bf_size(params).second / group_size;
    for (size_t candidate : {size_t{8}, size_t{4}, size_t{2}}) {
        if ((groups_k % candidate) == 0) {
            cfg.sg_k = candidate;
            break;
        }
    }

    return cfg;
}

// GEMM variants
// -------------
// Every FC builds the activation quantizer (kernel 0) plus a list of GEMM variants
// (kernels 1..); exactly one GEMM runs per inference. The whole kernel requires
// supports_immad (see Validate). The variants are:
//
//   scalar  - no DPAS. One row per subgroup, sg_k (8/4/2) subgroups split K and
//             reduce through SLM. Always the last entry; runs for rows < 8
//             (dpas_min_batch), i.e. decode.
//   DPAS v1 - tile_m rows (8, 16 or 32) per subgroup; sg_m subgroups of a
//             workgroup share each decoded weight group through SLM. Any zero
//             point layout, any arch with immad.
//   DPAS v2 - 16 x 64 tiles per subgroup (nb = 4 column blocks), 2D block reads,
//             256 GRF, sg_m 8 or 16. Needs Xe2+, a scalar or absent zero point,
//             N % 64 == 0, K >= 64 and an even number of granules per group
//             (supports_v2).
//
// Which DPAS variants are compiled depends on whether the row count is known:
//
//   static shape            - one DPAS config from get_dpas_config: tile_m 8 for
//                             rows <= 8, 16 for rows <= 16, otherwise v2 (sg_m 8 from
//                             96 rows, 16 from 192) if supported, else v1 32-row
//                             tiles with sg_m from get_dense_sg_m.
//   dynamic, dense variants - four DPAS configs keyed by min_rows: v1 32x sg_m 1
//                             (0 rows) and sg_m 2 (48), then v2 sg_m 8 (96) and 16
//                             (192) if supported, else v1 sg_m 4 (96) and 8 (384).
//                             Used when every sg_m is valid for the group layout
//                             (use_dense_variants).
//   dynamic, otherwise      - a single v1 32 x 1 config.
//
// select_gemm then picks, from the runtime row count, scalar below dpas_min_batch
// and otherwise the DPAS entry with the largest min_rows not above it.
std::vector<gemm_config> get_gemm_configs(const fully_connected_params& params, bool dense_variants) {
    std::vector<gemm_config> configs;
    if (dense_variants) {
        configs = get_dense_variants(params);
    } else {
        configs.push_back(get_dpas_config(params));
    }
    configs.push_back(get_scalar_config(params));
    return configs;
}

// Index into get_gemm_configs of the variant to run for these params.
size_t select_gemm(const fully_connected_params& params, const std::vector<gemm_config>& configs) {
    const size_t rows = get_input_bf_size(params).first;
    if (rows < dpas_min_batch)
        return configs.size() - 1;
    size_t best = 0;
    for (size_t i = 0; i + 1 < configs.size(); ++i) {
        if (configs[i].min_rows <= rows && configs[i].min_rows >= configs[best].min_rows)
            best = i;
    }
    return best;
}

// One subgroup per quantization group, several subgroups per workgroup where the
// group count allows it.
CommonDispatchData get_quantize_dispatch(size_t num_groups) {
    CommonDispatchData dispatchData;
    num_groups = std::max(num_groups, size_t{1});
    size_t sgs_per_wg = 1;
    for (size_t candidate : {size_t{16}, size_t{8}, size_t{4}, size_t{2}}) {
        if ((num_groups % candidate) == 0) {
            sgs_per_wg = candidate;
            break;
        }
    }
    dispatchData.gws = {num_groups * simd, 1, 1};
    dispatchData.lws = {sgs_per_wg * simd, 1, 1};
    return dispatchData;
}

CommonDispatchData get_gemm_dispatch(const fully_connected_params& params, const gemm_config& cfg) {
    CommonDispatchData dispatchData;

    const size_t output_f = get_output_aligned_bf_size(params, false).second;
    const size_t n_blocks = CeilDiv(output_f, osv);
    const size_t rows = std::max(get_input_bf_size(params).first, size_t{1});

    if (cfg.dpas) {
        const size_t m_groups = CeilDiv(rows, cfg.tile_m * cfg.sg_m);
        dispatchData.gws = {(n_blocks / cfg.nb) * simd, m_groups * cfg.sg_m, 1};
        dispatchData.lws = {simd, cfg.sg_m, 1};
    } else {
        dispatchData.gws = {n_blocks * simd, CeilDiv(rows, cfg.tile_m) * cfg.sg_k, 1};
        dispatchData.lws = {simd, cfg.sg_k, 1};
    }

    return dispatchData;
}

}  // namespace

ParamsKey FullyConnected_int3_dpas::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::UINT3);
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
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

DeviceFeaturesKey FullyConnected_int3_dpas::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_short();
    return k;
}

bool FullyConnected_int3_dpas::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto& input = fc_params.inputs[0];
    const auto& output = fc_params.outputs[0];
    const auto& weights = fc_params.weights;

    // The matrix engine is the whole point of this kernel; without it the generic
    // kernels are a better choice.
    if (!fc_params.engineInfo.supports_immad) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (!fc_params.compressed || weights.GetDType() != WeightsType::UINT3) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (input.GetDType() != Datatype::F16) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (input.GetFirstElementOffset() != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 ||
        input.Batch().pad.Total() != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (output.GetLayout() == DataLayout::bfyx && input.X().v > 1) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // The weights reorder produces whole (16 output x 32 input) blocks; anything
    // that does not fill them exactly would need edge handling the GEMM lacks.
    const size_t ifm = get_input_bf_size(fc_params).second;
    const size_t ofm = get_output_aligned_bf_size(fc_params, false).second;
    if (ifm == 0 || ofm == 0 || weights.IFM().v != ifm || weights.OFM().v != ofm) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    if ((ifm % k_chunk) != 0 || (ofm % osv) != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Rows of the quantized activation buffer are read with uint / block_read_us4,
    // both of which need the row stride to stay 4-byte aligned.
    if ((ifm % 4) != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // The quantizer walks the activation tensor as one flat run and the GEMM
    // addresses it by row stride, so the two only agree when the stride is the
    // row length. That also keeps the per-group scale index (row * var_pitch + g)
    // exact.
    if (get_input_b_pitch(fc_params) != ifm) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const size_t group_size = get_quantize_group_size(fc_params);
    if (group_size < k_chunk || (group_size % k_chunk) != 0 || (ifm % group_size) != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // The weight scale, and the weight zero point when there is one, have to be
    // constant across a dynamic quantization group: the group is the unit at which
    // the integer accumulator is drained and rescaled.
    const size_t scale_group_size = get_wei_scale_group_size(fc_params);
    if (scale_group_size < group_size || (scale_group_size % group_size) != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    if (!is_addressable_dtype(fc_params.decompression_scale.GetDType())) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (fc_params.has_decompression_zp && !fc_params.scalar_zp) {
        const auto zp_groups = fc_params.decompression_zero_point.Feature().v;
        if (zp_groups == 0) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        const size_t zp_group_size = weights.IFM().v / zp_groups;
        if (zp_group_size < group_size || (zp_group_size % group_size) != 0) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        if (!is_addressable_dtype(fc_params.decompression_zero_point.GetDType())) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }

    return true;
}

JitConstants FullyConnected_int3_dpas::GetJitConstants(const fully_connected_params& params,
                                                      const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    const size_t group_size = get_quantize_group_size(params);
    jit.AddConstant(MakeJitConstant("QUANTIZE_GROUP_SIZE", group_size));
    jit.AddConstant(MakeJitConstant("IFM_SIZE", get_input_bf_size(params).second));
    // The scale [N, groups] is addressed through its own pitches, per channel and group.
    const auto& scale = params.decompression_scale;
    jit.AddConstant(MakeJitConstant("WEI_SCALE_OFFSET", scale.GetFirstElementOffset()));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_N_PITCH", scale.Batch().pitch));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_G_PITCH", scale.Feature().pitch));
    jit.AddConstant(MakeJitConstant("WEI_SCALE_GROUP_SIZE", get_wei_scale_group_size(params)));

    const auto activation_dt = Datatype::F32;
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", get_input_b_pitch(params)));
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    return jit;
}

JitConstants FullyConnected_int3_dpas::GetGemmJitConstants(const fully_connected_params& params,
                                                          const gemm_config& cfg) const {
    // The launch geometry lives in gemm_config, not DispatchData, so GetJitConstants
    // has nothing to read out of it.
    JitConstants jit = GetJitConstants(params, DispatchData());

    jit.AddConstant(MakeJitConstant("USE_DPAS", cfg.dpas ? 1 : 0));
    jit.AddConstant(MakeJitConstant("DPAS_V2", cfg.v2 ? 1 : 0));
    jit.AddConstant(MakeJitConstant("V2_NB", cfg.nb));
    jit.AddConstant(MakeJitConstant("TILE_M", cfg.tile_m));
    jit.AddConstant(MakeJitConstant("SG_M", cfg.sg_m));
    jit.AddConstant(MakeJitConstant("SG_K", cfg.sg_k));

    // The store addresses output element (out_row, n), out_row being the row of
    // the flattened batch. A 3D bfyx output [B, M, N] splits it back into b and f.
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "out_row", "n", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx)
            idx_order = { "out_row / OUTPUT_FEATURE_NUM", "out_row % OUTPUT_FEATURE_NUM", "n", "0" };
        FusedOpsConfiguration conf = { "", idx_order, "activated", Datatype::F32, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnected_int3_dpas::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto configs = get_gemm_configs(fc_params, use_dense_variants(fc_params));

    KernelData kd = KernelData::Default<fully_connected_params>(params, 1 + configs.size());
    auto& new_params = *static_cast<fully_connected_params*>(kd.params.get());

    if (!UpdateWeightsParams(new_params, WeightsLayout::os_is_yx_osv16_isv32, kd.weightsReorderParams, GetSupportedKey()))
        return {};

    const size_t group_size = get_quantize_group_size(new_params);
    OPENVINO_ASSERT(group_size != 0, "[GPU] int3 FC: dynamic quantization group size is zero.");
    const size_t input_size = get_quantized_input_size(fc_params);
    const size_t var_size = (input_size / group_size) * 2 * sizeof(float);

    int inputs_count = 2;  // input + decompression scale
    if (new_params.has_decompression_zp && !new_params.scalar_zp)
        inputs_count++;

    // Kernel 0: activation quantizer.
    {
        auto& quan_kernel = kd.kernels[0];
        const auto quan_dispatch = get_quantize_dispatch(input_size / group_size);

        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, 0);
        auto cldnn_jit = GetJitConstants(new_params, DispatchData());
        cldnn_jit.AddConstant(MakeJitConstant("FC_KERNEL_DYNAMIC_QUANTIZE", 1));
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        FillCLKernelData(quan_kernel,
                         quan_dispatch,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         EXE_MODE_DEFAULT,
                         false,
                         false,
                         1,
                         0,
                         0,
                         fc_params.is_shape_agnostic);

        quan_kernel.params.arguments.clear();
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        quan_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        quan_kernel.skip_execution = false;
    }

    kd.internalBuffers.push_back(input_size);
    kd.internalBuffers.push_back(var_size);
    kd.internalBufferDataType = Datatype::F16;

    // Kernels 1..: the GEMM variants. Only one of them runs per inference.
    const size_t selected = select_gemm(fc_params, configs);

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& cfg = configs[i];
        auto& gemm_kernel = kd.kernels[i + 1];
        const auto dispatch = get_gemm_dispatch(fc_params, cfg);

        auto entry_point = GetEntryPoint(kernelName, fc_params.layerID, params, static_cast<int>(i) + 1);
        auto jit = CreateJit(kernelName, GetGemmJitConstants(new_params, cfg), entry_point);

        FillCLKernelData(gemm_kernel,
                         dispatch,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         EXE_MODE_DEFAULT,
                         true,
                         !fc_params.bias.empty(),
                         inputs_count,
                         GetFusedPrimitiveInputsCount(params),
                         1,
                         fc_params.is_shape_agnostic);

        gemm_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        gemm_kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        gemm_kernel.skip_execution = (i != selected);
        if (cfg.v2)
            add_large_grf_option(gemm_kernel);
    }

    GetUpdateDispatchDataFunc(kd);

    return {kd};
}

void FullyConnected_int3_dpas::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const fully_connected_params&>(params);

        const size_t group_size = get_quantize_group_size(prim_params);
        OPENVINO_ASSERT(group_size != 0, "[GPU] int3 FC: dynamic quantization group size is zero.");

        const size_t input_size = get_quantized_input_size(prim_params);
        const size_t var_size = (input_size / group_size) * 2 * sizeof(float);
        if (kd.internalBuffers[0].byte_count < input_size || kd.internalBuffers[1].byte_count < var_size) {
            kd.internalBuffers.clear();
            kd.internalBuffers.push_back(input_size);
            kd.internalBuffers.push_back(var_size);
        }

        const bool skip = KernelData::SkipKernelExecution(prim_params);

        const auto quan_dispatch = get_quantize_dispatch(input_size / group_size);
        kd.kernels[0].params.workGroups.global = quan_dispatch.gws;
        kd.kernels[0].params.workGroups.local = quan_dispatch.lws;
        kd.kernels[0].skip_execution = skip;

        // The runtime params are always built as shape-agnostic, so the variant set
        // compiled into kd is recognised by its kernel count instead.
        const bool dense_variants = kd.kernels.size() == 2 + dense_variant_count;
        const auto configs = get_gemm_configs(prim_params, dense_variants);
        OPENVINO_ASSERT(kd.kernels.size() == 1 + configs.size(), "[GPU] int3 FC: unexpected kernel count.");
        const size_t selected = select_gemm(prim_params, configs);
        for (size_t i = 0; i < configs.size(); ++i) {
            auto& kernel = kd.kernels[i + 1];
            kernel.skip_execution = skip || i != selected;
            if (kernel.skip_execution)
                continue;
            const auto dispatch = get_gemm_dispatch(prim_params, configs[i]);
            kernel.params.workGroups.global = dispatch.gws;
            kernel.params.workGroups.local = dispatch.lws;
        }
    };
}

KernelsPriority FullyConnected_int3_dpas::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
