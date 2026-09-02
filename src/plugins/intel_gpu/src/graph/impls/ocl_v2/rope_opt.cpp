// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "rope_opt.hpp"

#include <sstream>

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

size_t get_vec_size(const RuntimeParams& params) {
    const auto& input = params.get_input_layout(0);
    const auto& input1 = params.get_input_layout(1);
    auto desc = params.typed_desc<rope>();
    size_t vec_size = 1;
    switch (input.data_type) {
    case ov::element::f16:
        vec_size = 16;
        break;
    case ov::element::f32:
        vec_size = 8;
        break;
    default:
        vec_size = 1;
        break;
    }
    if (desc->config.rotary_ndims % (2 * vec_size) != 0) {
        vec_size = 1;
    }

    // Some models use f32 precision for input1 (cos) and input2 (sin) for better accuracy.
    // If input0 is not f32, we set vec_size as 1 for simple type conversion.
    if (input1.data_type == ov::element::f32 && input.data_type != input1.data_type)
        vec_size = 1;

    if (desc->config.is_qwen) {
        auto count = desc->config.head_cnt * std::max(desc->config.rotary_ndims / 2ul, desc->config.head_size - desc->config.rotary_ndims);
        if (count % vec_size != 0) {
            vec_size = 1;
        }
    }

    // Debug hook: OV_ROPE_VEC forces the per-work-item vector width. The kernel only has
    // bodies for 1, 8 and 16, and rotary_ndims must stay divisible by 2*vec_size.
    if (const char* s = std::getenv("OV_ROPE_VEC")) {
        size_t v = static_cast<size_t>(std::atoi(s));
        if ((v == 1 || v == 8 || v == 16) && desc->config.rotary_ndims % (2 * v) == 0) {
            vec_size = v;
        }
    }

    return vec_size;
}

// The interleaved dispatch normally puts BATCH on gws dim 0, which is the largest stride of a
// bfyx tensor, so a subgroup's lanes land on unrelated cache lines. Reversing dims 0 and 2 puts
// the rotary/head index -- the contiguous axis -- on the fast dimension instead.
static bool interleaved_reversed_gws(const RuntimeParams& params) {
    if (std::getenv("OV_ROPE_NO_REVERSE") != nullptr) {
        return false;
    }
    auto desc = params.typed_desc<rope>();
    const auto& cfg = desc->config;
    return cfg.is_interleaved && !cfg.is_qwen && !cfg.is_chatglm && !cfg.is_ltx_video && !cfg.support_3d_rope;
}

// Upstream reverses the rotate-half dispatch only at vec_size 1, so a vectorised rotate-half
// keeps BATCH on the fast dimension and its subgroups straddle whole tensors. The reversal is
// just as valid vectorised: the kernel already reads b from gws dim 2 under REVERSED_GWS.
static bool half_reversed_gws(const RuntimeParams& params, size_t vec_size) {
    if (std::getenv("OV_ROPE_NO_HALF_REVERSE") != nullptr) {
        return false;
    }
    auto desc = params.typed_desc<rope>();
    const auto& cfg = desc->config;
    return !cfg.is_interleaved && vec_size > 1 && !cfg.is_qwen && !cfg.is_chatglm && !cfg.is_ltx_video && !cfg.support_3d_rope;
}

// A packed bfyx tensor whose rows are HEAD_COUNT * ROTARY_NDIMS wide can be walked as one flat
// run of VEC_SIZE-element chunks, which is the only lane->address mapping the compiler lowers to
// a subgroup block read. Returns 0 when that is not legal here, otherwise the flat gws.
//
// Measured neutral against the default REVERSED_GWS path, so it is off unless explicitly asked for.
// OV_ROPE_CONTIG: unset/0 = off, 1 = on, "copy" = on but drop the rotation (bandwidth ablation).
static size_t interleaved_contig_gws(const RuntimeParams& params, size_t vec_size) {
    const char* mode = std::getenv("OV_ROPE_CONTIG");
    if (mode == nullptr || std::string(mode) == "0") {
        return 0;
    }
    auto desc = params.typed_desc<rope>();
    const auto& cfg = desc->config;
    if (!cfg.is_interleaved || cfg.is_qwen || cfg.is_chatglm || cfg.is_ltx_video || cfg.support_3d_rope) {
        return 0;
    }
    if (cfg.input_trans0213 || cfg.output_trans0213 || desc->gather_rank > 0 || cfg.slice_stop > cfg.slice_start) {
        return 0;
    }
    if (vec_size != 8 && vec_size != 16) {
        return 0;
    }
    const size_t rot = cfg.rotary_ndims;
    if (rot != cfg.head_size || rot == 0 || (rot & (rot - 1)) != 0 || rot % (2 * vec_size) != 0) {
        return 0;
    }
    const auto& in_l = params.input_layouts[0];
    const auto& cos_l = params.input_layouts[1];
    const auto& sin_l = params.input_layouts[2];
    const auto& out_l = params.output_layouts[0];
    for (const auto* l : {&in_l, &cos_l, &sin_l, &out_l}) {
        if (l->format != format::bfyx || l->count() != l->get_linear_size() || l->data_padding.is_dynamic()) {
            return 0;
        }
    }
    if (in_l.data_type != cos_l.data_type || in_l.data_type != sin_l.data_type ||
        (in_l.data_type != out_l.data_type && out_l.data_type != ov::element::i8)) {
        return 0;
    }
    // input0 is (b, tokens, heads, rot); cos/sin are (b, tokens, 1, rot), so a cos element is at
    // flat_token * rot + (elem % rot).
    if (extract_channel(ChannelName::X, in_l) != rot || extract_channel(ChannelName::Y, in_l) != cfg.head_cnt) {
        return 0;
    }
    for (const auto* l : {&cos_l, &sin_l}) {
        if (extract_channel(ChannelName::X, *l) != rot || extract_channel(ChannelName::Y, *l) != 1 ||
            extract_channel(ChannelName::BATCH, *l) != extract_channel(ChannelName::BATCH, in_l) ||
            extract_channel(ChannelName::FEATURE, *l) != extract_channel(ChannelName::FEATURE, in_l)) {
            return 0;
        }
    }
    if (out_l.count() != in_l.count() || in_l.count() % vec_size != 0) {
        return 0;
    }
    return in_l.count() / vec_size;
}

class RopeGenerator : public KernelGenerator {
public:
    RopeGenerator() : KernelGenerator("rope_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<rope>();

        auto in_l = params.input_layouts[0];
        jit.make("HEAD_SIZE", desc->config.head_size);
        jit.make("ROTARY_NDIMS", desc->config.rotary_ndims);
        jit.make("HALF_ROTARY_NDIMS", desc->config.rotary_ndims / 2);
        jit.make("COS_SIN_TABLE_OFFSET", (desc->config.cos_sin_ndims == (desc->config.rotary_ndims / 2)) ? 0 : desc->config.rotary_ndims / 2);
        jit.make("HEAD_COUNT", desc->config.head_cnt);

        if (desc->config.head_size > desc->config.rotary_ndims) {
            jit.make("ENABLE_IO_COPY", true);
        }

        if (desc->gather_rank > 0) {
            jit.make("ENABLE_GATHER", true);
            jit.make("GATHER_RANK", desc->gather_rank);
        }

        if (desc->config.slice_stop > desc->config.slice_start) {
            jit.make("ENABLE_SLICE", true);
            jit.make("SLICED_FROM_START", to_code_string(desc->config.slice_start));
        }

        if (desc->config.input_trans0213) {
            jit.make("ENABLE_TRANSPOSE", true);
        }

        if (!desc->config.is_chatglm && (params.input_layouts[1].data_padding.is_dynamic() || params.input_layouts[2].data_padding.is_dynamic())) {
            jit.make("SIN_COS_HAVE_DYNAMIC_PADDINGS", true);
        }

        if (desc->config.is_qwen) {
            jit.make("QWEN", true);
        } else if (desc->config.is_chatglm) {
            if (desc->config.support_2d_rope) {
                jit.make("SUPPORT_2D_ROPE", true);
            }
            if (desc->config.use_rope_cache) {
                jit.make("USE_ROPE_CACHE", true);
            }
            jit.make("CHATGLM", true);
        } else if (desc->config.is_ltx_video) {
            jit.make("LTX_VIDEO", true);
        } else if (desc->config.is_interleaved) {
            if (!params.is_dynamic() && interleaved_contig_gws(params, get_vec_size(params)) != 0) {
                jit.make("ROPE_CONTIG", true);
                const char* mode = std::getenv("OV_ROPE_CONTIG");
                if (mode != nullptr && std::string(mode) == "copy") {
                    jit.make("ROPE_CONTIG_COPY", true);
                }
            } else {
                jit.make("RotateInterleaved", true);
                if (interleaved_reversed_gws(params)) {
                    jit.make("REVERSED_GWS", true);
                }
            }
        } else {
            jit.make("RotateHalf", true);
            if (get_vec_size(params) == 1 || half_reversed_gws(params, get_vec_size(params))) {
                jit.make("REVERSED_GWS", true);
            }
        }
        jit.make("VEC_SIZE", get_vec_size(params));
        if (params.get_output_layout(0).data_type == ov::element::i8) {
            jit.make("OUTPUT_I8", true);
        }
        if (params.get_input_layout(0).data_type != params.get_input_layout(1).data_type) {
            jit.add(make_type_jit_constants("ACCUMULATOR", params.get_input_layout(1).data_type));
        } else {
            jit.add(make_type_jit_constants("ACCUMULATOR", params.get_input_layout(0).data_type));
        }
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        auto desc = params.typed_desc<rope>();
        uint32_t num_of_inputs =
            (desc->config.is_chatglm && desc->config.use_rope_cache) || (desc->config.output_trans0213 && desc->config.is_interleaved) ? 2 : 3;

        if (desc->gather_rank > 0) {
            num_of_inputs++;
        }

        for (uint32_t i = 0; i < num_of_inputs; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
                size_t vec_size = get_vec_size(params);
                auto desc = params.typed_desc<rope>();
                const auto& cfg = desc->config;
                std::vector<std::vector<ChannelName>> dims_by_gws = {{ChannelName::BATCH}, {ChannelName::FEATURE}, {ChannelName::Y, ChannelName::X}};
                const auto& in_l = params.input_layouts[0];
                const auto& out_l = params.output_layouts[0];

                auto largest_divisor_leq = [](size_t n, size_t cap) {
                    for (size_t d = std::min(n, cap); d > 1; d--) {
                        if (n % d == 0) {
                            return d;
                        }
                    }
                    return size_t{1};
                };

                if (size_t flat = interleaved_contig_gws(params, vec_size); flat != 0) {
                    wgs.global = {flat, 1, 1};
                    const size_t cap = std::min<size_t>(256, static_cast<size_t>(params.get_device_info().max_work_group_size));
                    // Prefer a whole number of 16-wide subgroups; fall back to any divisor.
                    size_t l0 = 1;
                    for (size_t d = (cap / 16) * 16; d >= 16; d -= 16) {
                        if (flat % d == 0) {
                            l0 = d;
                            break;
                        }
                    }
                    wgs.local = {l0 > 1 ? l0 : largest_divisor_leq(flat, cap), 1, 1};
                    if (const char* s = std::getenv("OV_ROPE_LWS0")) {
                        size_t v = static_cast<size_t>(std::atoi(s));
                        if (v > 0 && flat % v == 0) {
                            wgs.local = {v, 1, 1};
                        }
                    }
                    return;
                }

                if (cfg.is_qwen) {
                    auto b = extract_channel(ChannelName::BATCH, in_l);
                    auto f = extract_channel(ChannelName::FEATURE, in_l);
                    wgs.global = {b, f, cfg.head_cnt * std::max(cfg.rotary_ndims / 2ul, cfg.head_size - cfg.rotary_ndims) / vec_size};
                } else if (cfg.is_chatglm) {
                    auto b = extract_channel(ChannelName::BATCH, in_l);
                    auto f = extract_channel(ChannelName::FEATURE, in_l);

                    if (cfg.support_2d_rope) {
                        // input  [batch_size, seq_length]
                        // output [batch_size, head_count, seq_length, half_rotary_ndims]
                        wgs.global = {b * cfg.head_cnt, f, cfg.rotary_ndims / 2ul / vec_size};
                    } else {
                        wgs.global = {b, f, cfg.head_cnt * (cfg.rotary_ndims / 2ul) / vec_size};
                    }
                } else if (cfg.is_ltx_video) {
                    auto b = extract_channel(ChannelName::BATCH, in_l);
                    auto f = extract_channel(ChannelName::FEATURE, in_l);
                    wgs.global = {b, f, cfg.rotary_ndims / 2ul / vec_size};
                } else {
                    auto b = extract_channel(ChannelName::BATCH, out_l);
                    auto f = extract_channel(ChannelName::FEATURE, out_l);
                    auto y = extract_channel(ChannelName::Y, out_l);
                    wgs.global = {b, f, y * cfg.rotary_ndims / 2ul / vec_size};
                    if (cfg.support_3d_rope) {
                        wgs.global = {b, f, cfg.rotary_ndims / 2ul / vec_size};
                    }
                    // reverse gws when RotateHalf and vec_size is one
                    if (!desc->config.is_interleaved && vec_size == 1) {
                        size_t tmp = wgs.global[0];
                        wgs.global[0] = wgs.global[2];
                        wgs.global[2] = tmp;
                    }
                }

                auto largest_divisor = [](size_t n, size_t cap) {
                    for (size_t d = std::min(n, cap); d > 1; d--) {
                        if (n % d == 0) {
                            return d;
                        }
                    }
                    return size_t{1};
                };

                if (half_reversed_gws(params, vec_size)) {
                    std::swap(wgs.global[0], wgs.global[2]);
                    // gws0 is a multiple of the rotary half-width here, so a whole number of
                    // 16-wide subgroups fits and every lane stays inside one row.
                    const size_t l0 = wgs.global[0] % 32 == 0 ? 32 : (wgs.global[0] % 16 == 0 ? 16 : 1);
                    wgs.local = {l0, largest_divisor(wgs.global[1], std::max(size_t{1}, 256 / l0)), 1};
                    return;
                }

                if (interleaved_reversed_gws(params)) {
                    std::swap(wgs.global[0], wgs.global[2]);
                    // Keep dim 0 whole where possible so a subgroup stays inside one row, then
                    // spend what is left of the workgroup budget on the sequence dimension.
                    const size_t l0 = largest_divisor(wgs.global[0], 32);
                    wgs.local = {l0, largest_divisor(wgs.global[1], std::max(size_t{1}, 256 / l0)), 1};
                    // Debug hook: OV_ROPE_LWS="a b c" forces the local size, so the workgroup
                    // shape can be swept without a rebuild. gws0 is 18 here, which is not a
                    // multiple of the 16-wide subgroup, so the default straddles rows.
                    if (const char* s = std::getenv("OV_ROPE_LWS")) {
                        std::istringstream is(s);
                        size_t a = 0, b2 = 0, c = 0;
                        if (is >> a >> b2 >> c && a && b2 && c && wgs.global[0] % a == 0 && wgs.global[1] % b2 == 0 &&
                            wgs.global[2] % c == 0) {
                            wgs.local = {a, b2, c};
                        }
                    }
                    return;
                }

                // We need to set the 1st local workgroup size as large as possible for better performance.
                if (vec_size == 1) {
                    auto get_max_lws = [](size_t gws, size_t max_workgroup_size) -> size_t {
                        size_t val = 1;
                        size_t lws = 1;
                        while (((val + 1) <= max_workgroup_size) && (gws >= (val + 1))) {
                            val += 1;
                            if (gws % val == 0)
                                lws = val;
                        }
                        return lws;
                    };

                    size_t max_workgroup_size = static_cast<size_t>(params.get_device_info().max_work_group_size);

                    wgs.local = {1, 1, 1};
                    wgs.local[0] = get_max_lws(wgs.global[0], max_workgroup_size);
                    max_workgroup_size /= wgs.local[0];
                    wgs.local[1] = get_max_lws(wgs.global[1], max_workgroup_size);
                    max_workgroup_size /= wgs.local[1];
                    wgs.local[2] = get_max_lws(wgs.global[2], max_workgroup_size);
                } else {
                    wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
                }
            }
        }};
    }
};

class RopeOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::RopeOptImpl)

    Stage::Ptr rope = make_stage<RopeGenerator>();

    RopeOptImpl() : PrimitiveImplOCL(RopeOpt::get_type_info_static()) {}
    RopeOptImpl(const program_node& node, const RuntimeParams& params) : RopeOptImpl() {
        add_stage(rope, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<RopeOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> RopeOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<rope>());
    return std::make_unique<RopeOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rope)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::RopeOptImpl)
