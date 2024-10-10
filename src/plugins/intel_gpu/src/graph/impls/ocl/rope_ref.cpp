// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "rope_ref.hpp"
#include "impls/ocl/jitter.hpp"
#include "impls/ocl/kernel_base.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_ocl_base.hpp"

namespace cldnn {
namespace ocl {

namespace {

using namespace ov::intel_gpu::ocl;

class RopeGeneratorRef : public ov::intel_gpu::ocl::SingleKernelGenerator {
public:
    RopeGeneratorRef() : SingleKernelGenerator("rope_ref") {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit = SingleKernelGenerator::get_jit_constants(node, params);
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

            const auto axis =  desc->config.is_qwen || desc->config.is_chatglm ? 2 : 3;

            LayoutJitter in_layout(in_l, 0);

            auto f = in_layout.dim(ChannelName::FEATURE);
            auto x = in_layout.dim(ChannelName::X);
            auto y = in_layout.dim(ChannelName::Y);

            auto sliced_val = to_code_string(desc->config.slice_stop - desc->config.slice_start);
            auto sliced_x = axis == 3 ? sliced_val : x;
            auto sliced_y = axis == 2 ? sliced_val : y;

            jit.make("SLICED_INPUT0_X_PITCH", 1);
            jit.make("SLICED_INPUT0_Y_PITCH", sliced_x);
            jit.make("SLICED_INPUT0_FEATURE_PITCH", sliced_x + "*" + sliced_y);
            jit.make("SLICED_INPUT0_BATCH_PITCH", sliced_x + "*" + sliced_y + "*" + f);
            jit.make("SLICED_INPUT0_OFFSET", 0);
            jit.make("SLICED_FROM_START", to_code_string(desc->config.slice_start));

            if (axis == 2) {
                jit.make("SLICED_FROM_END", "(" + y + "-" + to_code_string(desc->config.slice_stop) + ")");
            } else if (axis == 3) {
                jit.make("SLICED_FROM_END", "(" + x + "-" + to_code_string(desc->config.slice_stop) + ")");
            } else {
                OPENVINO_THROW("[GPU] Invalid axis value for RoPE operation");
            }
        }

        if (desc->config.input_trans0213) {
            jit.make("ENABLE_TRANSPOSE", true);
            jit.make("TRANSPOSED_INPUT0_OFFSET", 0);
            jit.make("TRANSPOSED_INPUT0_X_PITCH", 1);
            jit.make("TRANSPOSED_INPUT0_Y_PITCH", "INPUT0_FEATURE_PITCH");
            jit.make("TRANSPOSED_INPUT0_FEATURE_PITCH", "INPUT0_Y_PITCH");
            jit.make("TRANSPOSED_INPUT0_BATCH_PITCH", "INPUT0_BATCH_PITCH");
        }

        if (!desc->config.is_chatglm && (params.input_layouts[1].data_padding.is_dynamic()|| params.input_layouts[2].data_padding.is_dynamic())) {
            jit.make("SIN_COS_HAVE_DYNAMIC_PADDINGS", true);
        }

        if (desc->config.is_qwen) {
            jit.make("QWEN", true);
        } else if (desc->config.is_chatglm) {
            jit.make("CHATGLM", true);
        } else {
            jit.make("RotateHalf", true);
        }
        return jit;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        auto desc = params.typed_desc<rope>();
        uint32_t num_of_inputs = desc->config.is_chatglm || desc->config.is_interleaved ? 2 : 3;

        if (desc->gather_rank > 0) {
            num_of_inputs++;
        }

        for (uint32_t i = 0; i < num_of_inputs; i++)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes dispatch_data;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<rope>();
                const auto& cfg = desc->config;
                // std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH }, { Tensor::DataChannelName::FEATURE },
                //                                                                 { Tensor::DataChannelName::Y, Tensor::DataChannelName::X }};
                if (cfg.is_chatglm || cfg.is_qwen) {
                    const auto& in_l = params.input_layouts[0];
                    auto b = extract_channel(ChannelName::BATCH, in_l);
                    auto f = extract_channel(ChannelName::FEATURE, in_l);
                    dispatch_data.global = {b, f, cfg.head_cnt * std::max(cfg.rotary_ndims / 2ul, cfg.head_size - cfg.rotary_ndims)};
                } else {
                    const auto& out_l = params.output_layouts[0];
                    auto b = extract_channel(ChannelName::BATCH, out_l);
                    auto f = extract_channel(ChannelName::FEATURE, out_l);
                    auto y = extract_channel(ChannelName::Y, out_l);

                    dispatch_data.global = {b, f, y * cfg.rotary_ndims / 2ul};
                }

                dispatch_data.local = {1, 1, 1};
                // dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, input.GetLayout(), output.GetLayout(), dims_by_gws);
            }

            return { dispatch_data, {} };
        };
        return f;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> RopeRef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<rope>());
    RopeGeneratorRef gen;
    return cldnn::make_unique<primitive_impl_ocl>(gen.get_kernels_data(node, params), std::string(get_type_info().name));
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rope)
