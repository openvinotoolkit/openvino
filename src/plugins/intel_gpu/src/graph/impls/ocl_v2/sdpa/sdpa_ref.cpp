// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sdpa_ref.hpp"

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "kv_cache_inst.h"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"

namespace ov::intel_gpu::ocl {
namespace {

ov::element::Type get_accumulator_type(const kernel_impl_params& params) {
    return params.get_input_layout(0).data_type;
}

class SDPARefGenerator : public SDPABase {
public:
    explicit SDPARefGenerator(bool indirect) : SDPABase("sdpa_ref", "", indirect) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SDPABase::get_jit_constants(params);
        jit.add(make_tensors_jit_constants(params));
        auto desc = params.typed_desc<scaled_dot_product_attention>();

        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(params)));
        if (desc->has_sink_input) {
            jit.make("HAS_SINK_INPUT", 1);
            jit.make("SINK_DATA_T", "half");
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        auto desc = params.typed_desc<scaled_dot_product_attention>();

        size_t data_inputs_num = desc->input_size();
        if (desc->indirect_axis != -1) {
            data_inputs_num--;
        }

        auto has_zp_input_buffers = desc->get_compression_zp_inputs_num() > 0;
        if (desc->is_kv_compressed) {
            data_inputs_num -= 2;  // key and value compression scales are handled separately

            if (desc->get_compression_zp_inputs_num() > 0) {
                data_inputs_num -= 2;  // key and value compression zp are handled separately
            }
        }

        for (uint32_t i = 0; i < data_inputs_num; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        auto beam_table_idx = data_inputs_num;
        if (desc->is_kv_compressed) {
            auto key_cache_compression_scale_idx = static_cast<uint32_t>(data_inputs_num);
            auto value_cache_compression_scale_idx = static_cast<uint32_t>(data_inputs_num + 1);

            args.push_back({ArgumentDescriptor::Types::INPUT, key_cache_compression_scale_idx});
            args.push_back({ArgumentDescriptor::Types::INPUT, value_cache_compression_scale_idx});

            if (has_zp_input_buffers) {
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

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<scaled_dot_product_attention>();

                const auto& out_l = params.output_layouts[0];
                auto b = extract_channel(ChannelName::BATCH, out_l);
                auto f = extract_channel(ChannelName::FEATURE, out_l);
                auto y = extract_channel(ChannelName::Y, out_l);
                auto x = extract_channel(ChannelName::X, out_l);

                wgs.global = {b * f, y, x};
                wgs.local = {1, 1, x};
            }
        }};
    }
};

}  // namespace

class SDPARefImpl : public SDPAImplBase {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SDPARefImpl)
    Stage::Ptr indirect = make_stage<SDPARefGenerator>(true);
    Stage::Ptr regular = make_stage<SDPARefGenerator>(false);

    SDPARefImpl() : SDPAImplBase(SDPARef::get_type_info_static()) {}
    explicit SDPARefImpl(const RuntimeParams& params) : SDPARefImpl() {
        add_stage(regular, params);
        add_stage(indirect, params);
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        update_rt_params(instance);

        if (need_indirect_load(static_cast<scaled_dot_product_attention_inst&>(instance))) {
            return execute_stage(events, instance, indirect);
        }
        return execute_stage(events, instance, regular);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        std::vector<BufferDescriptor> internal_buffers;

        auto desc = params.typed_desc<scaled_dot_product_attention>();
        const auto& q_l = params.input_layouts[0];
        if (!params.is_dynamic()) {
            const auto& k_l = params.input_layouts[1];

            const auto& q_shape = q_l.get_shape();
            const auto& k_shape = k_l.get_shape();
            const size_t buf_size = q_l.count() / q_shape[desc->input_q_transpose_order[3]] * k_shape[desc->input_k_transpose_order[2]];
            internal_buffers.emplace_back(buf_size, q_l.data_type);
        } else {
            internal_buffers.emplace_back(1, q_l.data_type);
        }

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SDPARefImpl>(this);
    }
};

std::unique_ptr<primitive_impl> SDPARef::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPARefImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

// BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SDPARefImpl)
