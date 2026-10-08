// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_int3_dpas.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "common_utils/jitter.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/serialization/weights_reorder_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr const char* kernel_name = "fully_connected_int3_dpas";
constexpr size_t simd = 16;
constexpr size_t k_chunk = 32;  // u3 values per granule, and the DPAS K step
constexpr size_t osv = 16;      // output channels per weights block
// Rows at or above which the matrix-engine variants win over the K-split one, which reads all weights once per row.
constexpr size_t dpas_min_rows = 2;
// Column blocks per subgroup and rows per subgroup of the v2 path. Its 16 x 64 tiles need 256 registers per thread.
constexpr size_t v2_nb = 4;
constexpr size_t v2_tile_m = 16;
// Row counts from which v2 takes over with 8 and with 16 subgroups per workgroup. A workgroup covers 16 * sg_m rows,
// so each step pays off once it is mostly full. Below 96 rows the 32-row tiles of v1 are as fast or faster.
constexpr size_t v2_min_rows_sg8 = 96;
constexpr size_t v2_min_rows_sg16 = 192;

size_t get_scale_idx(const fully_connected& desc) {
    return desc.bias.is_valid() ? 3 : 2;
}

// Product of all dimensions but the last, i.e. the rows of the flattened batch.
size_t get_rows(const layout& input) {
    const auto shape = input.get_shape();
    return std::accumulate(shape.begin(), shape.end() - 1, size_t{1}, std::multiplies<>());
}

// The decompression scale and zero point are read as whole elements, so a packed type would index the wrong value.
bool is_addressable_dtype(ov::element::Type dt) {
    return one_of(dt, {ov::element::f16, ov::element::f32, ov::element::i8, ov::element::u8, ov::element::i32, ov::element::u32});
}

bool is_dense(const layout& l) {
    return !l.data_padding && !l.data_padding.is_dynamic();
}

// A static, unpadded [N, groups] tensor (trailing dims of 1) in a plain format. Constant transposes turn the scale
// and zero point into fbyx, so they are addressed through their pitches rather than as bfyx.
bool get_n_by_groups_pitches(const layout& l, size_t& groups, size_t& n_pitch, size_t& g_pitch) {
    if (l.is_dynamic() || !is_dense(l) || !format::is_simple_data_format(l.format)) {
        return false;
    }
    const auto shape = l.get_shape();
    if (shape.size() < 2 || std::any_of(shape.begin() + 2, shape.end(), [](size_t d) {
            return d != 1;
        })) {
        return false;
    }
    const auto pitches = l.get_pitches();
    groups = shape[1];
    n_pitch = static_cast<size_t>(pitches[0]);
    g_pitch = static_cast<size_t>(pitches[1]);
    return true;
}

// Everything the kernel needs to know about an FC apart from its row count.
struct Int3FcInfo {
    size_t ifm = 0;
    size_t ofm = 0;
    size_t scale_group_size = 0;
    size_t scale_n_pitch = 0;
    size_t scale_g_pitch = 0;
    bool has_zp = false;
    bool scalar_zp = false;
    float zp_value = 0.0f;
    size_t zp_batch_num = 0;
    size_t zp_batch_pitch = 0;
    size_t zp_feature_pitch = 0;
    size_t zp_group_size = 0;
    // Dynamic quantization group of the activations.
    size_t group_size = 0;
};

// Fills info and returns true when the kernel supports the FC. in_layouts are the FC's input layouts: activations,
// weights, optional bias, scale and optional zero point.
bool get_int3_fc_info(const fully_connected& desc, const std::vector<layout>& in_layouts, size_t dyn_quan_group_size, Int3FcInfo& info) {
    const size_t scale_idx = get_scale_idx(desc);
    if (!desc.decompression_scale.is_valid() || in_layouts.size() <= scale_idx) {
        return false;
    }

    const auto& in_pshape = in_layouts[0].get_partial_shape();
    const auto& w_pshape = in_layouts[1].get_partial_shape();
    if (in_pshape.size() < 2 || in_pshape.size() > 3 || in_pshape[in_pshape.size() - 1].is_dynamic()) {
        return false;
    }
    if (w_pshape.is_dynamic() || w_pshape.size() < 2) {
        return false;
    }
    for (size_t i = 2; i < w_pshape.size(); ++i) {
        if (w_pshape[i].get_length() != 1) {
            return false;
        }
    }

    info.ifm = static_cast<size_t>(in_pshape[in_pshape.size() - 1].get_length());
    info.ofm = static_cast<size_t>(w_pshape[0].get_length());
    // The weights reorder produces whole (16 output x 32 input) blocks; anything that does not fill them exactly
    // would need edge handling the GEMM lacks. K % 32 also keeps the quantized activation rows 4-byte aligned.
    if (static_cast<size_t>(w_pshape[1].get_length()) != info.ifm || info.ifm == 0 || info.ofm == 0 || info.ifm % k_chunk != 0 || info.ofm % osv != 0) {
        return false;
    }

    // The scale is [N, groups]: one row of K groups per output channel.
    const auto& scale = in_layouts[scale_idx];
    size_t scale_groups = 0;
    if (!is_addressable_dtype(scale.data_type) || !get_n_by_groups_pitches(scale, scale_groups, info.scale_n_pitch, info.scale_g_pitch)) {
        return false;
    }
    if (scale.get_shape()[0] != info.ofm || scale_groups == 0 || info.ifm % scale_groups != 0) {
        return false;
    }
    info.scale_group_size = info.ifm / scale_groups;

    if (desc.decompression_zero_point.is_valid()) {
        if (in_layouts.size() <= scale_idx + 1) {
            return false;
        }
        const auto& zp = in_layouts[scale_idx + 1];
        if (zp.is_dynamic() || !is_dense(zp)) {
            return false;
        }
        info.has_zp = true;
        if (zp.count() == 1 && desc.decompression_zero_point_scalar.has_value()) {
            info.scalar_zp = true;
            info.zp_value = desc.decompression_zero_point_scalar.value();
        } else {
            size_t zp_groups = 0;
            if (!is_addressable_dtype(zp.data_type) || !get_n_by_groups_pitches(zp, zp_groups, info.zp_batch_pitch, info.zp_feature_pitch)) {
                return false;
            }
            if (zp_groups == 0 || info.ifm % zp_groups != 0) {
                return false;
            }
            info.zp_batch_num = zp.get_shape()[0];
            info.zp_group_size = info.ifm / zp_groups;
        }
    } else if (desc.decompression_zero_point_scalar.has_value()) {
        info.has_zp = true;
        info.scalar_zp = true;
        info.zp_value = desc.decompression_zero_point_scalar.value();
    }

    // Deliberately not bf_tiled's dynamic quantization group: the group is the unit the DPAS path decodes into
    // registers at once and stages through SLM, so it has to stay small and bounded. It is also the unit at which
    // the integer accumulator is drained and rescaled, so the weight scale - and the zero point, which is folded in
    // via the activation sum - must be constant across it.
    for (size_t candidate : {size_t{128}, size_t{64}, size_t{32}}) {
        if (dyn_quan_group_size < candidate || info.ifm % candidate != 0 || info.scale_group_size % candidate != 0) {
            continue;
        }
        if (info.zp_group_size != 0 && info.zp_group_size % candidate != 0) {
            continue;
        }
        info.group_size = candidate;
        return true;
    }

    return false;
}

Int3FcInfo get_int3_fc_info(const RuntimeParams& params) {
    Int3FcInfo info;
    const auto dyn_quan_group_size = params.get_program().get_config().get_dynamic_quantization_group_size();
    OPENVINO_ASSERT(get_int3_fc_info(*params.typed_desc<fully_connected>(), params.input_layouts, dyn_quan_group_size, info),
                    "[GPU] ",
                    params.desc->id,
                    ": unsupported configuration for ",
                    kernel_name);
    return info;
}

struct GemmConfig {
    bool dpas = false;
    bool v2 = false;
    size_t tile_m = 1;
    size_t sg_m = 1;
    size_t nb = 1;
};

// The sg_m subgroups split one staging iteration's granules between them, so the iteration has to cover whole
// quantization groups and split evenly.
bool is_valid_sg_m(const Int3FcInfo& info, size_t sg_m) {
    const size_t chunks_per_group = info.group_size / k_chunk;
    const size_t groups_k = info.ifm / info.group_size;
    const size_t groups_per_iter = (sg_m > chunks_per_group) ? sg_m / chunks_per_group : 1;
    const size_t chunks_per_iter = groups_per_iter * chunks_per_group;
    return (chunks_per_iter % sg_m) == 0 && (groups_k % groups_per_iter) == 0;
}

// Subgroups sharing one weight decode through SLM, by row count, with 32-row tiles. Device-timed on the Qwen3-8B
// shapes (N and K from 4096 to 12288): each doubling of sg_m pays off once the workgroup's 32 * sg_m rows are mostly
// filled, up to ~21 TOPS at sg_m 8 versus ~10 at 1.
size_t get_dense_sg_m(size_t rows) {
    if (rows >= 384) {
        return 8;
    }
    if (rows >= 96) {
        return 4;
    }
    if (rows >= 48) {
        return 2;
    }
    return 1;
}

// The v2 path relies on 2D block reads and split work-group barriers (Xe2 and later), shares one decode across nb
// column blocks and folds a scalar zero point into a per-row term, so it is limited to FCs whose zero point is
// scalar or absent and whose output channels fill whole nb * 16 column tiles.
bool supports_v2(const Int3FcInfo& info, gpu_arch arch) {
    if (arch < gpu_arch::xe2) {
        return false;
    }
    if (info.has_zp && !info.scalar_zp) {
        return false;
    }
    if (info.group_size < 2 * k_chunk || ((info.group_size / k_chunk) % 2) != 0) {
        return false;
    }
    if ((info.ofm % (v2_nb * osv)) != 0) {
        return false;
    }
    // 2D block reads need a surface width and pitch of at least 64 bytes.
    return info.ifm >= 64;
}

bool is_valid_v2_sg_m(const Int3FcInfo& info, size_t sg_m) {
    const size_t gran_per_group = (info.group_size / k_chunk) * v2_nb;
    const size_t groups_k = info.ifm / info.group_size;
    const size_t groups_per_iter = (sg_m > 2 * gran_per_group) ? sg_m / gran_per_group : 2;
    return ((groups_per_iter * gran_per_group) % sg_m) == 0 && (groups_per_iter % 2) == 0 && (groups_k % groups_per_iter) == 0;
}

GemmConfig get_v1_config(size_t tile_m, size_t sg_m) {
    return GemmConfig{true, false, tile_m, sg_m, 1};
}

GemmConfig get_v2_config(size_t sg_m) {
    return GemmConfig{true, true, v2_tile_m, sg_m, v2_nb};
}

// DPAS variant for a row count: tile_m 8 up to 8 rows, 16 up to 16 rows, otherwise v2 (sg_m 8 from 96 rows, 16 from
// 192) if supported, else v1 32-row tiles with sg_m from get_dense_sg_m.
GemmConfig get_dpas_config(const Int3FcInfo& info, gpu_arch arch, size_t rows) {
    if (rows <= 8) {
        return get_v1_config(8, 1);
    }
    if (rows <= 16) {
        return get_v1_config(16, 1);
    }
    if (rows >= v2_min_rows_sg8 && supports_v2(info, arch)) {
        const size_t sg_m = (rows >= v2_min_rows_sg16) ? 16 : 8;
        if (is_valid_v2_sg_m(info, sg_m)) {
            return get_v2_config(sg_m);
        }
    }
    for (size_t sg_m = get_dense_sg_m(rows); sg_m > 1; sg_m /= 2) {
        if (is_valid_sg_m(info, sg_m)) {
            return get_v1_config(32, sg_m);
        }
    }
    return get_v1_config(32, 1);
}

// Subgroups splitting K in the scalar path.
size_t get_scalar_sg_k(const Int3FcInfo& info) {
    const size_t groups_k = info.ifm / info.group_size;
    for (size_t candidate : {size_t{8}, size_t{4}, size_t{2}}) {
        if ((groups_k % candidate) == 0) {
            return candidate;
        }
    }
    return 1;
}

std::shared_ptr<WeightsReorderParams> make_weights_reorder_params(const layout& weights) {
    if (weights.format == format::os_is_yx_osv16_isv32) {
        return nullptr;
    }
    const auto& pshape = weights.get_partial_shape();
    const layout repacked(ov::PartialShape{pshape[0], pshape[1]}, weights.data_type, format::os_is_yx_osv16_isv32);
    return std::make_shared<WeightsReorderParams>(weights, repacked, false);
}

class Int3FcGeneratorBase : public KernelGenerator {
public:
    explicit Int3FcGeneratorBase(std::string_view suffix) : KernelGenerator(kernel_name, suffix) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        const auto info = get_int3_fc_info(params);
        auto jit = make_base_jit_constants(params);
        jit.add(make_type_jit_constants("INPUT0", params.get_input_layout(0).data_type));
        jit.add(make_type_jit_constants("OUTPUT", params.get_output_layout(0).data_type));
        jit.make("QUANTIZE_GROUP_SIZE", info.group_size);
        return jit;
    }
};

class Int3FcQuantize : public Int3FcGeneratorBase {
public:
    Int3FcQuantize() : Int3FcGeneratorBase("quantize") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = Int3FcGeneratorBase::get_jit_constants(params);
        jit.make("FC_KERNEL_DYNAMIC_QUANTIZE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {{ArgumentDescriptor::Types::INPUT, 0}, {ArgumentDescriptor::Types::INTERNAL_BUFFER, 0}, {ArgumentDescriptor::Types::INTERNAL_BUFFER, 1}};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto info = get_int3_fc_info(params);
            // One subgroup per quantization group, several subgroups per workgroup where the group count allows it.
            const size_t num_groups = std::max<size_t>(get_rows(params.get_input_layout(0)) * info.ifm / info.group_size, 1);
            size_t sgs_per_wg = 1;
            for (size_t candidate : {size_t{16}, size_t{8}, size_t{4}, size_t{2}}) {
                if ((num_groups % candidate) == 0) {
                    sgs_per_wg = candidate;
                    break;
                }
            }
            kd.params.workGroups.global = {num_groups * simd, 1, 1};
            kd.params.workGroups.local = {sgs_per_wg * simd, 1, 1};
        }};
    }
};

// One GEMM variant. Its JIT depends on N, K, the group layout, the zero point kind and the fused ops, but not on the
// row count, which arrives as a kernel argument.
class Int3FcGemm : public Int3FcGeneratorBase {
public:
    Int3FcGemm(std::string_view suffix, GemmConfig cfg) : Int3FcGeneratorBase(suffix), m_cfg(cfg) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        const auto info = get_int3_fc_info(params);
        const auto& desc = *params.typed_desc<fully_connected>();
        const size_t scale_idx = get_scale_idx(desc);
        auto jit = Int3FcGeneratorBase::get_jit_constants(params);

        jit.make("IFM_SIZE", info.ifm);
        jit.make("TILE_IN_B_PITCH", info.ifm);
        jit.make("TILE_OUT_F_NUM", info.ofm);
        jit.make("TILE_OUT_F_PITCH", 1);
        jit.make("TILE_OUT_B_PITCH", info.ofm);
        jit.make("OUTPUT_OFFSET", 0);
        // A constant lets the compiler fold the row bounds checks around the loads and stores
        if (params.is_dynamic()) {
            jit.make("BATCH_SIZE", "rows");
        } else {
            jit.make("BATCH_SIZE", get_rows(params.get_input_layout(0)));
        }

        jit.make("FILTER_TYPE", "uchar");
        jit.make("DECOMPRESSION_SCALE_TYPE", to_ocl_type(params.get_input_layout(scale_idx).data_type));
        // The scale [N, groups] is addressed per channel and group.
        jit.make("WEI_SCALE_OFFSET", 0);
        jit.make("WEI_SCALE_N_PITCH", info.scale_n_pitch);
        jit.make("WEI_SCALE_G_PITCH", info.scale_g_pitch);
        jit.make("WEI_SCALE_GROUP_SIZE", info.scale_group_size);

        jit.make("DECOMPRESSION_ZP_TERM", info.has_zp);
        jit.make("DECOMPRESSION_ZP_SCALAR", info.scalar_zp);
        if (info.has_zp) {
            if (info.scalar_zp) {
                jit.make("DECOMPRESSION_ZP_VALUE", info.zp_value);
            } else {
                jit.make("DECOMPRESSION_ZP_TYPE", to_ocl_type(params.get_input_layout(scale_idx + 1).data_type));
                jit.make("DECOMPRESSION_ZP_BATCH_NUM", info.zp_batch_num);
                jit.make("DECOMPRESSION_ZP_BATCH_PITCH", info.zp_batch_pitch);
                jit.make("DECOMPRESSION_ZP_FEATURE_PITCH", info.zp_feature_pitch);
                jit.make("DECOMPRESSION_ZP_GROUP_SIZE", info.zp_group_size);
            }
        }

        jit.make("BIAS_TERM", desc.bias.is_valid());
        if (desc.bias.is_valid()) {
            jit.make("BIAS_TYPE", to_ocl_type(params.get_input_layout(2).data_type));
        }

        jit.make("USE_DPAS", m_cfg.dpas);
        jit.make("DPAS_V2", m_cfg.v2);
        jit.make("V2_NB", m_cfg.nb);
        jit.make("TILE_M", m_cfg.tile_m);
        jit.make("SG_M", m_cfg.sg_m);
        jit.make("SG_K", m_cfg.dpas ? 1 : get_scalar_sg_k(info));

        // The store addresses output element (out_row, n), out_row being the row of the flattened batch. A 3D bfyx
        // output [B, M, N] splits it back into b and f. The fused ops compute this index per element, so M is a
        // constant wherever it is static.
        if (params.has_fused_primitives()) {
            std::vector<std::string> idx_order = {"out_row", "n", "0", "0"};
            const auto& out_pshape = params.get_output_layout(0).get_partial_shape();
            if (out_pshape.size() == 3) {
                if (out_pshape[0].is_static() && out_pshape[0].get_length() == 1) {
                    idx_order = {"0", "out_row", "n", "0"};
                } else {
                    const std::string m = out_pshape[1].is_static() ? std::to_string(out_pshape[1].get_length()) : "rows_per_batch";
                    idx_order = {"(out_row / " + m + ")", "(out_row % " + m + ")", "n", "0"};
                }
            }
            FusedOpsConfiguration conf = {"", idx_order, "activated", ov::element::f32, 1};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto info = get_int3_fc_info(params);
        const auto& desc = *params.typed_desc<fully_connected>();

        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        if (info.has_zp && !info.scalar_zp) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
        if (desc.bias.is_valid()) {
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
        }
        add_fused_ops_arguments(args, params);
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        return args;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = KernelGenerator::get_build_options(params);
        if (m_cfg.v2) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[cfg = m_cfg](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            const auto info = get_int3_fc_info(params);
            const auto& input = params.get_input_layout(0);
            const size_t rows = get_rows(input);
            const size_t dispatch_rows = std::max<size_t>(rows, 1);
            const size_t n_blocks = info.ofm / osv;

            auto& wgs = kd.params.workGroups;
            if (cfg.dpas) {
                const size_t m_groups = ceil_div(dispatch_rows, cfg.tile_m * cfg.sg_m);
                wgs.global = {(n_blocks / cfg.nb) * simd, m_groups * cfg.sg_m, 1};
                wgs.local = {simd, cfg.sg_m, 1};
            } else {
                const size_t sg_k = get_scalar_sg_k(info);
                wgs.global = {n_blocks * simd, dispatch_rows * sg_k, 1};
                wgs.local = {simd, sg_k, 1};
            }

            const size_t rows_per_batch = input.get_partial_shape().size() == 3 ? input.get_shape()[1] : rows;
            kd.params.scalars.clear();
            for (size_t v : {rows, rows_per_batch}) {
                scalar_desc s;
                s.t = scalar_desc::Types::UINT32;
                s.v.u32 = static_cast<uint32_t>(v);
                kd.params.scalars.push_back(s);
            }
        }};
    }

private:
    GemmConfig m_cfg;
};

// GEMM variants, indexed like FullyConnectedInt3DpasImpl::gemms.
enum class GemmVariant : uint8_t { scalar, v1_t8, v1_t16, v1_t32_sg1, v1_t32_sg2, v1_t32_sg4, v1_t32_sg8, v2_sg8, v2_sg16 };

GemmVariant get_gemm_variant(const Int3FcInfo& info, gpu_arch arch, size_t rows) {
    if (rows < dpas_min_rows) {
        return GemmVariant::scalar;
    }
    const auto cfg = get_dpas_config(info, arch, rows);
    if (cfg.v2) {
        return cfg.sg_m == 16 ? GemmVariant::v2_sg16 : GemmVariant::v2_sg8;
    }
    if (cfg.tile_m == 8) {
        return GemmVariant::v1_t8;
    }
    if (cfg.tile_m == 16) {
        return GemmVariant::v1_t16;
    }
    switch (cfg.sg_m) {
    case 8:
        return GemmVariant::v1_t32_sg8;
    case 4:
        return GemmVariant::v1_t32_sg4;
    case 2:
        return GemmVariant::v1_t32_sg2;
    default:
        return GemmVariant::v1_t32_sg1;
    }
}

class FullyConnectedInt3DpasImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FullyConnectedInt3DpasImpl)

    // Every variant is declared so that an impl loaded from the cache finds the stages it was built with. A static
    // impl activates the quantizer and the variant for its row count. A dynamic impl activates every variant the FC
    // can use, so that they are all compiled with the model, and picks one per execution.
    Stage::Ptr quantize = make_stage<Int3FcQuantize>();
    std::array<Stage::Ptr, 9> gemms = {make_stage<Int3FcGemm>("scalar", GemmConfig{}),
                                       make_stage<Int3FcGemm>("v1_t8", get_v1_config(8, 1)),
                                       make_stage<Int3FcGemm>("v1_t16", get_v1_config(16, 1)),
                                       make_stage<Int3FcGemm>("v1_t32_sg1", get_v1_config(32, 1)),
                                       make_stage<Int3FcGemm>("v1_t32_sg2", get_v1_config(32, 2)),
                                       make_stage<Int3FcGemm>("v1_t32_sg4", get_v1_config(32, 4)),
                                       make_stage<Int3FcGemm>("v1_t32_sg8", get_v1_config(32, 8)),
                                       make_stage<Int3FcGemm>("v2_sg8", get_v2_config(8)),
                                       make_stage<Int3FcGemm>("v2_sg16", get_v2_config(16))};

    FullyConnectedInt3DpasImpl() : PrimitiveImplOCL(FullyConnectedInt3Dpas::get_type_info_static()) {}
    FullyConnectedInt3DpasImpl(const program_node& node, const RuntimeParams& params) : FullyConnectedInt3DpasImpl() {
        const auto info = get_int3_fc_info(params);
        const auto arch = params.get_device_info().arch;

        add_stage(quantize, params);
        if (params.is_dynamic()) {
            // get_gemm_variant is constant over each of these row counts' ranges
            for (size_t rows : {size_t{1}, size_t{8}, size_t{16}, size_t{17}, size_t{48}, v2_min_rows_sg8, v2_min_rows_sg16, size_t{384}}) {
                add_gemm_stage(get_gemm_variant(info, arch, rows), params);
            }
        } else {
            add_gemm_stage(get_gemm_variant(info, arch, get_rows(params.get_input_layout(0))), params);
        }

        if (params.weights_layout.has_value()) {
            _weights_reorder_params = make_weights_reorder_params(params.weights_layout.value());
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FullyConnectedInt3DpasImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto info = get_int3_fc_info(params);
        const size_t input_size = std::max<size_t>(get_rows(params.get_input_layout(0)) * info.ifm, info.group_size);
        return {BufferDescriptor{input_size, ov::element::i8}, BufferDescriptor{(input_size / info.group_size) * 2, ov::element::f32}};
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::kernel_impl_params& impl_params) const override {
        if (_order.size() <= 2) {
            return _order;
        }
        const auto variant = get_gemm_variant(get_int3_fc_info(impl_params), impl_params.get_device_info().arch, get_rows(impl_params.get_input_layout(0)));
        const size_t gemm_idx = get_stage_idx(gemms[static_cast<size_t>(variant)]);
        OPENVINO_ASSERT(is_activated(gemm_idx), "[GPU] ", kernel_name, ": GEMM variant ", static_cast<size_t>(variant), " is not compiled");
        return {_order[0], gemm_idx};
    }

protected:
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        auto args = PrimitiveImplOCL::get_arguments(instance);
        const auto& fc_inst = static_cast<const fully_connected_inst&>(instance);
        const auto& desc = *instance.get_typed_desc<fully_connected>();
        size_t scale_idx = get_scale_idx(desc);

        args.inputs = {fc_inst.input_memory_ptr(0), fc_inst.dep_memory_ptr(scale_idx)};
        if (desc.decompression_zero_point.is_valid()) {
            args.inputs.push_back(fc_inst.dep_memory_ptr(scale_idx + 1));
        }
        args.weights = fc_inst.weights_memory();
        args.bias = fc_inst.bias_term() ? fc_inst.bias_memory() : nullptr;
        return args;
    }

private:
    void add_gemm_stage(GemmVariant variant, const RuntimeParams& params) {
        auto& stage = gemms[static_cast<size_t>(variant)];
        if (!is_activated(get_stage_idx(stage))) {
            add_stage(stage, params);
        }
    }

    [[nodiscard]] size_t get_stage_idx(const Stage::Ptr& stage) const {
        return static_cast<size_t>(std::distance(_stages.begin(), std::find(_stages.begin(), _stages.end(), stage.get())));
    }

    [[nodiscard]] bool is_activated(size_t stage_idx) const {
        return std::find(_order.begin(), _order.end(), stage_idx) != _order.end();
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FullyConnectedInt3Dpas::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FullyConnectedInt3DpasImpl>(node, params);
}

bool FullyConnectedInt3Dpas::validate_impl(const program_node& node) const {
    assert(node.is_type<fully_connected>());
    const auto& fc_node = node.as<fully_connected>();
    const auto& desc = *fc_node.get_primitive();
    const auto& program = node.get_program();
    const auto& config = program.get_config();

    // The matrix engine is the whole point of this kernel; without it the generic kernels are a better choice.
    if (!program.get_engine().get_device_info().supports_immad) {
        return false;
    }

    const auto& forcing = config.get_force_implementations();
    const auto forced = forcing.find(node.id());
    if (forced != forcing.end() && !forced->second.kernel_name.empty() && forced->second.kernel_name != kernel_name) {
        return false;
    }

    if (!desc.compressed_weights || !desc.weights_transposed || desc.weights_rank != 2 ||
        fc_node.weights().get_output_layout(false).data_type != ov::element::u3) {
        return false;
    }

    const auto& in_layout = fc_node.get_input_layout(0);
    const auto& out_layout = fc_node.get_output_layout(0);
    if (in_layout.data_type != ov::element::f16 || !one_of(out_layout.data_type, {ov::element::f16, ov::element::f32})) {
        return false;
    }
    if (!one_of(in_layout.format, {format::bfyx, format::any}) || !one_of(out_layout.format, {format::bfyx, format::any})) {
        return false;
    }
    // The quantizer walks the activations as one flat run and the GEMM addresses them by row stride, so the two only
    // agree for dense activations.
    if (!is_dense(in_layout) || !is_dense(out_layout) || in_layout.get_partial_shape().size() != out_layout.get_partial_shape().size()) {
        return false;
    }

    if (desc.bias.is_valid() && !is_dense(fc_node.bias().get_output_layout(false))) {
        return false;
    }

    if (!fused_ops_are_one_of<eltwise, activation, quantize>(node.get_fused_primitives())) {
        return false;
    }

    Int3FcInfo info;
    return get_int3_fc_info(desc, node.get_input_layouts(), config.get_dynamic_quantization_group_size(), info);
}

bool FullyConnectedInt3Dpas::support_shapes(const RuntimeParams& params) const {
    return is_dense(params.get_input_layout(0)) && is_dense(params.get_output_layout(0));
}

in_out_fmts_t FullyConnectedInt3Dpas::query_formats(const program_node& node) const {
    assert(node.is_type<fully_connected>());
    std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
    std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

    const size_t out_rank = node.get_output_layout().get_rank();
    for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
        if (!node.get_dependency(idx).is_constant()) {
            in_fmts[idx] = format::get_default_format(out_rank);
        }
    }
    out_fmts[0] = format::get_default_format(out_rank);

    return {in_fmts, out_fmts};
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FullyConnectedInt3DpasImpl)
