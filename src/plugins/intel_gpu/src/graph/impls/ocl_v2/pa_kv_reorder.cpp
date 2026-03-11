// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {
constexpr size_t subgroup_size = 16;
class PaKVReorderGenerator : public KernelGenerator {
public:
    PaKVReorderGenerator() : KernelGenerator("pa_kv_cache_reorder_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;

        constexpr static std::array input_ids = {cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES,
                                                 cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES_BEGINS,
                                                 cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES,
                                                 cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES_BEGINS};

        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids.at(i);
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        constexpr size_t key_cache_id = cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE;
        constexpr size_t value_cache_id = cldnn::pa_kv_reorder::PaKVReorderInputIdx::VALUE_CACHE;

        jit.add(make_layout_jit_constants("OUTPUT", params.input_layouts[key_cache_id], in_offsets_map.at(key_cache_id)));
        jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(1), params.input_layouts[value_cache_id], in_offsets_map.at(value_cache_id)));

        const auto desc = params.typed_desc<cldnn::pa_kv_reorder>();
        OPENVINO_ASSERT(desc->kv_heads_num > 0, "[GPU] pa_kv_reorder expects positive kv_heads_num in descriptor");
        OPENVINO_ASSERT(desc->adjusted_k_head_size > 0 && desc->adjusted_v_head_size > 0, "[GPU] pa_kv_reorder expects positive head sizes in descriptor");
        OPENVINO_ASSERT(desc->adjusted_paged_attention_block_size > 0, "[GPU] pa_kv_reorder expects positive block size in descriptor");

        const size_t kv_heads_num = desc->kv_heads_num;
        const size_t adjusted_k_head_size = desc->adjusted_k_head_size;
        const size_t adjusted_paged_attention_block_size = desc->adjusted_paged_attention_block_size;
        const size_t adjusted_v_head_size = desc->adjusted_v_head_size;

        jit.make("KV_HEADS_NUM", kv_heads_num);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", cldnn::paged_attention::block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);

        const auto key_cache_dt = desc->cache_dt;
        const bool is_kv_compressed = desc->is_kv_compressed;
        jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
        const size_t scales_zp_size = desc->scales_zp_size;
        const bool is_key_by_channel = desc->is_key_by_channel;

        if (is_kv_compressed) {
            if (is_key_by_channel) {
                jit.make("IS_KEY_BY_CHANNEL", 1);
                jit.make("K_HEAD_SIZE", adjusted_k_head_size);
                jit.make("ADJUSTED_K_HEAD_SIZE", adjusted_k_head_size);
                jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", adjusted_paged_attention_block_size);
            } else {
                OPENVINO_ASSERT(adjusted_k_head_size >= scales_zp_size, "[GPU] Invalid pa_kv_reorder key cache shape for compressed per-token mode");
                jit.make("K_HEAD_SIZE", adjusted_k_head_size - scales_zp_size);
                jit.make("ADJUSTED_K_HEAD_SIZE", adjusted_k_head_size);
                jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", cldnn::paged_attention::block_size);
            }
            OPENVINO_ASSERT(adjusted_v_head_size >= scales_zp_size, "[GPU] Invalid pa_kv_reorder value cache shape for compressed mode");
            jit.make("V_HEAD_SIZE", adjusted_v_head_size - scales_zp_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", adjusted_v_head_size);
        } else {
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", cldnn::paged_attention::block_size);
            jit.make("K_HEAD_SIZE", adjusted_k_head_size);
            jit.make("V_HEAD_SIZE", adjusted_v_head_size);
            jit.make("ADJUSTED_K_HEAD_SIZE", adjusted_k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", adjusted_v_head_size);
        }

        jit.add(make_type_jit_constants("UNCOMPRESSED", is_kv_compressed ? cldnn::data_types::f16 : key_cache_dt));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES_BEGINS});

        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, cldnn::pa_kv_reorder::PaKVReorderInputIdx::VALUE_CACHE});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.clear();

            const auto desc = params.typed_desc<cldnn::pa_kv_reorder>();
            OPENVINO_ASSERT(desc->kv_heads_num > 0, "[GPU] pa_kv_reorder expects positive kv_heads_num in descriptor");
            const auto heads_number = desc->kv_heads_num;

            const auto& begins_input = params.input_layouts[cldnn::pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES_BEGINS];
            const auto begins_len = static_cast<size_t>(begins_input.get_partial_shape()[0].get_length());
            OPENVINO_ASSERT(begins_len >= 1, "[GPU] BLOCK_UPDATE_INDICES_BEGINS must have at least 1 element");
            const auto sequences_number = begins_len - 1;

            const bool is_kv_compressed = desc->is_kv_compressed;

            auto wg_heads_number = std::min(heads_number, static_cast<size_t>((params.get_device_info().max_work_group_size) / heads_number));
            wgs.global = {sequences_number, heads_number, subgroup_size};
            wgs.local = {1, is_kv_compressed ? 1 : wg_heads_number, subgroup_size};
        }};
    }
};

class PaKVReorderImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PaKVReorderImpl)

    Stage::Ptr kv_reorder = make_stage<PaKVReorderGenerator>();

    PaKVReorderImpl() : PrimitiveImplOCL(PA_KV_reorder::get_type_info_static()) {}
    PaKVReorderImpl(const program_node& node, const RuntimeParams& params) : PaKVReorderImpl() {
        add_stage(kv_reorder, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PaKVReorderImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PA_KV_reorder::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<cldnn::pa_kv_reorder>());
    return std::make_unique<PaKVReorderImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::pa_kv_reorder)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PaKVReorderImpl)
