// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sdpa_ref.hpp"
#include "sdpa_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_base.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "primitive_ocl_base.hpp"
#include "kv_cache_inst.h"
#include "scaled_dot_product_attention_inst.h"

namespace ov::intel_gpu::ocl {

namespace {

ov::element::Type get_accumulator_type(const program_node& node) {
    return node.get_input_layout(0).data_type;
}

using namespace ov::intel_gpu::ocl;

class SDPARefGenerator : public SDPABase {
public:
    SDPARefGenerator(bool indirect) : SDPABase("sdpa_ref", indirect) {}

protected:
    JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const override {
        auto jit = SDPABase::get_jit_constants(node, params);
        auto desc = params.typed_desc<scaled_dot_product_attention>();

        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(node)));
        jit.make("NUM_KV_HEADS", -1);

        return jit;
    }

    Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        auto desc = params.typed_desc<scaled_dot_product_attention>();

        size_t inputs_count = desc->input_size();
        if (desc->indirect_axis != -1) {
            inputs_count--;
        }

        for (uint32_t i = 0; i < inputs_count; i++)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        auto beam_table_idx = inputs_count + 1;
        if (desc->is_kv_compressed) {
            auto key_cache_compression_scale_idx = static_cast<uint32_t>(desc->input_size());
            auto value_cache_compression_scale_idx = static_cast<uint32_t>(desc->input_size() + 1);

            args.push_back({ArgumentDescriptor::Types::INPUT, key_cache_compression_scale_idx});
            args.push_back({ArgumentDescriptor::Types::INPUT, value_cache_compression_scale_idx});

            const bool is_asym_quantization =
                desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
            const bool combined_scale_and_zp =
                desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

            if (is_asym_quantization && !combined_scale_and_zp) {
                args.push_back({ArgumentDescriptor::Types::INPUT, key_cache_compression_scale_idx + 2});
                args.push_back({ArgumentDescriptor::Types::INPUT, value_cache_compression_scale_idx + 2});
                beam_table_idx += 2;
            }

            beam_table_idx += 2;
        }

        if (m_indirect) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(beam_table_idx)});
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        return args;
    }

    DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const override {
        static auto f = DISPATCH_DATA_FUNC(params) {
            WorkGroupSizes wgs;
            std::vector<layout> intermediate_buffers;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<scaled_dot_product_attention>();

                const auto& out_l = params.output_layouts[0];
                auto b = extract_channel(ChannelName::BATCH, out_l);
                auto f = extract_channel(ChannelName::FEATURE, out_l);
                auto y = extract_channel(ChannelName::Y, out_l);
                auto x = extract_channel(ChannelName::X, out_l);

                wgs.global = { b * f, y, x };
                wgs.local = { 1, 1, x };
            }

            return { wgs, {} };
        };
        return f;
    }
};

}  // namespace

class SDPARefImpl : public PrimitiveImplOCL {
public:
    size_t INDIRECT_STAGE;
    size_t REGULAR_STAGE;

    SDPARefImpl(const program_node& node, const kernel_impl_params& params)
        : PrimitiveImplOCL(std::string(SDPARef::get_type_info_static().name)) {
        INDIRECT_STAGE = add_stage<SDPARefGenerator>(node, params, false);
        REGULAR_STAGE = add_stage<SDPARefGenerator>(node, params, false);
    }

    static size_t get_beam_table_id(std::shared_ptr<const scaled_dot_product_attention> primitive) {
        return primitive->input_size() - 1;
    }

    bool need_indirect_load(const scaled_dot_product_attention_inst& instance) const {
        auto desc = instance.get_typed_desc<scaled_dot_product_attention>();

        if (!instance.has_indirect_inputs())
            return false;

        const auto& params = *instance.get_impl_params();
        const auto indirect_axis = desc->indirect_axis;
        if (params.input_layouts[get_beam_table_id(desc)].get_partial_shape()[indirect_axis].get_length() == 1)
            return false;

        const auto& deps = instance.dependencies();

        const auto indirect_dep_idx = 1;
        const auto& indirect_dep = deps[indirect_dep_idx].first;
        if (dynamic_cast<const kv_cache_inst*>(indirect_dep) == nullptr) {
            return true;
        }

        auto state_layout = indirect_dep->get_impl_params()->get_input_layout(0);
        bool is_prefill = state_layout.count() == 0;
        return !is_prefill;
    }

    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& params) const override {
        std::vector<layout> bufs;

        const auto& q_l = params.input_layouts[0];
        if (!params.is_dynamic()) {
            const auto& k_l = params.input_layouts[1];

            const auto& q_shape = q_l.get_shape();
            const auto& k_shape = k_l.get_shape();
            const size_t buf_size = q_l.count() / q_shape[3] * k_shape[2];

            bufs = { layout{ov::PartialShape{static_cast<int64_t>(buf_size)}, q_l.data_type, format::bfyx } };
        } else {
            bufs = { layout{ov::PartialShape{1}, q_l.data_type, format::bfyx } };
        }

        return bufs;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<SDPARefImpl>(*this);
    }
};

std::unique_ptr<primitive_impl> SDPARef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPARefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
