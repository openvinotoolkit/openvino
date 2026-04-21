// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_causal_conv1d_ref.hpp"

#include "intel_gpu/primitives/paged_causal_conv1d.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

class PagedCausalConv1DRefGenerator : public KernelGenerator {
public:
    PagedCausalConv1DRefGenerator() : KernelGenerator("paged_causal_conv1d_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        const auto& input_shape = params.get_input_layout(paged_causal_conv1d::INPUT_EMBEDS).get_partial_shape();
        const auto& state_shape = params.get_input_layout(paged_causal_conv1d::CONV_STATE_TABLE).get_partial_shape();
        const auto& bias_shape = params.get_input_layout(paged_causal_conv1d::CONV_BIAS).get_partial_shape();
        const bool has_bias = bias_shape.rank().is_static() && bias_shape.size() == 1 && bias_shape[0].is_static() && bias_shape[0].get_length() != 0;

        jit.make("HIDDEN_SIZE", static_cast<int>(input_shape[1].get_length()));
        jit.make("KERNEL_SIZE", static_cast<int>(state_shape[2].get_length()));
        jit.make("HAS_BIAS", has_bias ? 1 : 0);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        constexpr size_t num_scalars = 13;  // Must match scalars count in get_dispatch_data_func
        for (size_t i = 0; i < num_scalars; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, static_cast<uint32_t>(i)});
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto& input_shape = params.get_input_layout(paged_causal_conv1d::INPUT_EMBEDS).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(paged_causal_conv1d::SUBSEQUENCE_BEGINS).get_partial_shape();

            const int32_t seq_count = seq_shape[0].get_length() > 0 ? static_cast<int32_t>(seq_shape[0].get_length() - 1) : 0;
            const int32_t hidden_size = static_cast<int32_t>(input_shape[1].get_length());

            auto read_pitch = [](const cldnn::layout& layout, size_t idx) -> int32_t {
                const auto& pitches = layout.get_pitches();
                if (idx < pitches.size()) {
                    return static_cast<int32_t>(pitches[idx]);
                }
                return 1;
            };

            const auto& in_layout = params.input_layouts[paged_causal_conv1d::INPUT_EMBEDS];
            const auto& state_layout = params.input_layouts[paged_causal_conv1d::CONV_STATE_TABLE];
            const auto& weight_layout = params.input_layouts[paged_causal_conv1d::CONV_WEIGHT];
            const auto& bias_layout = params.input_layouts[paged_causal_conv1d::CONV_BIAS];
            const auto& out_layout = params.output_layouts[0];
            const auto& bias_shape = bias_layout.get_partial_shape();
            const bool has_bias = bias_shape.rank().is_static() && bias_shape.size() == 1 && bias_shape[0].is_static() && bias_shape[0].get_length() != 0;

            const int32_t input_token_stride = read_pitch(in_layout, 0);
            const int32_t input_hidden_stride = read_pitch(in_layout, 1);

            const int32_t state_block_stride = read_pitch(state_layout, 0);
            const int32_t state_hidden_stride = read_pitch(state_layout, 1);
            const int32_t state_kernel_stride = read_pitch(state_layout, 2);

            const int32_t weight_hidden_stride = read_pitch(weight_layout, 0);
            const int32_t weight_kernel_stride = read_pitch(weight_layout, 2);

            const int32_t bias_hidden_stride = has_bias ? read_pitch(bias_layout, 0) : 0;

            const int32_t output_token_stride = read_pitch(out_layout, 0);
            const int32_t output_hidden_stride = read_pitch(out_layout, 1);

            wgs.global = {static_cast<size_t>(seq_count), static_cast<size_t>(hidden_size), 1};
            wgs.local = {1, 256, 1};
            if (wgs.local[1] > static_cast<size_t>(hidden_size)) {
                wgs.local[1] = static_cast<size_t>(hidden_size);
            }

            kd.params.scalars.clear();
            std::vector<int32_t> scalars{
                seq_count,
                hidden_size,
                input_token_stride,
                input_hidden_stride,
                state_block_stride,
                state_hidden_stride,
                state_kernel_stride,
                weight_hidden_stride,
                weight_kernel_stride,
                bias_hidden_stride,
                output_token_stride,
                output_hidden_stride,
                static_cast<int32_t>(params.get_input_layout(paged_causal_conv1d::CONV_STATE_TABLE).get_partial_shape()[0].get_length()),
            };

            for (auto v : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = v;
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class PagedCausalConv1DRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedCausalConv1DRefImpl)

    Stage::Ptr paged_causal_conv1d = make_stage<PagedCausalConv1DRefGenerator>();

    PagedCausalConv1DRefImpl() : PrimitiveImplOCL(PagedCausalConv1DRef::get_type_info_static()) {}
    PagedCausalConv1DRefImpl(const program_node& node, const RuntimeParams& params) : PagedCausalConv1DRefImpl() {
        add_stage(paged_causal_conv1d, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedCausalConv1DRefImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PagedCausalConv1DRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_causal_conv1d>());
    return std::make_unique<PagedCausalConv1DRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedCausalConv1DRefImpl)
