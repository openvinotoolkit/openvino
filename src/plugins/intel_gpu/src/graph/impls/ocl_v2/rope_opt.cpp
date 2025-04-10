// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "rope_opt.hpp"

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

size_t get_vec_size(const RuntimeParams& params) {
    const auto& input = params.get_input_layout(0);
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

    if (desc->config.is_qwen) {
        auto count = desc->config.head_cnt * std::max(desc->config.rotary_ndims / 2ul, desc->config.head_size - desc->config.rotary_ndims);
        if (count % vec_size != 0) {
            vec_size = 1;
        }
    }

    return vec_size;
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
            jit.make("CHATGLM", true);
        } else if (desc->config.is_interleaved) {
            jit.make("RotateInterleaved", true);
        } else {
            jit.make("RotateHalf", true);
        }
        jit.make("VEC_SIZE", get_vec_size(params));
        return jit;
    }

    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        auto desc = params.typed_desc<rope>();
        uint32_t num_of_inputs = desc->config.is_chatglm || (desc->config.output_trans0213 && desc->config.is_interleaved) ? 2 : 3;

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

                } else {
                    auto b = extract_channel(ChannelName::BATCH, out_l);
                    auto f = extract_channel(ChannelName::FEATURE, out_l);
                    auto y = extract_channel(ChannelName::Y, out_l);

                    wgs.global = {b, f, y * cfg.rotary_ndims / 2ul / vec_size};
                }

                wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), in_l.format, out_l.format, dims_by_gws);
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
