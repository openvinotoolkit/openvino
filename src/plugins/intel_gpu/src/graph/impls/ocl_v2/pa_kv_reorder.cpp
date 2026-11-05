// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {
constexpr size_t subgroup_size = 16;
constexpr size_t u4_elems_per_byte = 2;

// The K page layout of the cache in front of this reorder, from its PHYSICAL shape: EAGLE3's kv-update model has no
// PagedAttention op, so only the shape the main model created the cache with can tell (paged_attention.hpp,
// k_cache_token_major()). cache_dt is the configured precision (u4/i4 for INT4).
bool key_cache_token_major(const RuntimeParams& params) {
    const auto desc = params.typed_desc<cldnn::pa_kv_reorder>();
    if (desc->has_xattention) {
        return false;
    }
    const size_t block = cldnn::paged_attention::block_size;
    const bool int4 = data_type_traits::is_i4_u4(desc->cache_dt);
    const size_t page_block = (desc->is_kv_compressed && desc->is_key_by_channel) ? (int4 ? block / u4_elems_per_byte : block) + desc->scales_zp_size : block;
    return cldnn::paged_attention::k_cache_token_major(params.input_layouts[cldnn::pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE].get_partial_shape(),
                                                       page_block);
}

// Whether a token-major K page can reach this device at all: the layout rule needs k_token_major_device().
bool token_major_page_possible(const RuntimeParams& params) {
    using pa = cldnn::paged_attention;
    return pa::k_token_major_switch() != pa::k_layout_switch::d_major && pa::k_token_major_device(params.get_device_info());
}

class PaKVReorderGenerator : public KernelGenerator {
public:
    // One kernel per K page layout; the impl picks it from the cache shape at dispatch.
    explicit PaKVReorderGenerator(bool key_token_major)
        : KernelGenerator("pa_kv_cache_reorder_ref", key_token_major ? "tm" : ""),
          m_key_token_major(key_token_major) {}

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
        const bool is_int4_compressed = data_type_traits::is_i4_u4(key_cache_dt);
        jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
        jit.make("IS_INT4_COMPRESSED", is_int4_compressed ? 1 : 0);
        jit.make("U4_ELEMS_PER_BYTE", u4_elems_per_byte);
        const size_t scales_zp_size = desc->scales_zp_size;
        const bool is_key_by_channel = desc->is_key_by_channel;

        const auto align_to = [](size_t value, size_t alignment) {
            return ((value + alignment - 1) / alignment) * alignment;
        };

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
            const size_t k_head_size = is_key_by_channel ? adjusted_k_head_size : (adjusted_k_head_size - scales_zp_size);
            const size_t v_head_size = adjusted_v_head_size - scales_zp_size;
            jit.make("V_HEAD_SIZE", v_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", adjusted_v_head_size);

            if (is_int4_compressed) {
                size_t packed_k_head_size;
                size_t packed_adjusted_k_head_size;
                size_t packed_paged_attention_block_size;
                size_t packed_adjusted_paged_attention_block_size;
                if (is_key_by_channel) {
                    packed_k_head_size = k_head_size;
                    packed_adjusted_k_head_size = k_head_size;
                    packed_paged_attention_block_size = cldnn::paged_attention::block_size / u4_elems_per_byte;
                    packed_adjusted_paged_attention_block_size = packed_paged_attention_block_size + scales_zp_size;
                } else {
                    packed_k_head_size = align_to(k_head_size / u4_elems_per_byte, subgroup_size);
                    packed_adjusted_k_head_size = packed_k_head_size + scales_zp_size;
                    packed_paged_attention_block_size = cldnn::paged_attention::block_size;
                    packed_adjusted_paged_attention_block_size = cldnn::paged_attention::block_size;
                }
                const size_t packed_v_head_size = align_to(v_head_size / u4_elems_per_byte, subgroup_size);
                const size_t packed_adjusted_v_head_size = packed_v_head_size + scales_zp_size;
                jit.make("PACKED_K_HEAD_SIZE", packed_k_head_size);
                jit.make("PACKED_V_HEAD_SIZE", packed_v_head_size);
                jit.make("PACKED_ADJUSTED_K_HEAD_SIZE", packed_adjusted_k_head_size);
                jit.make("PACKED_ADJUSTED_V_HEAD_SIZE", packed_adjusted_v_head_size);
                jit.make("PACKED_PAGED_ATTENTION_BLOCK_SIZE", packed_paged_attention_block_size);
                jit.make("PACKED_ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", packed_adjusted_paged_attention_block_size);
            }
        } else {
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", cldnn::paged_attention::block_size);
            jit.make("K_HEAD_SIZE", adjusted_k_head_size);
            jit.make("V_HEAD_SIZE", adjusted_v_head_size);
            jit.make("ADJUSTED_K_HEAD_SIZE", adjusted_k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", adjusted_v_head_size);
        }

        // This variant's K page layout (IS_KEY_TOKEN_MAJOR covers every token-major page; the kernel splits it by the
        // quant mode). OV_GPU_PA_K_TM_BREAK=reorder compiles the token-major variant with the d-major read, to test
        // whether pa_kv_reorder_gpu actually observes it.
        bool key_token_major = m_key_token_major;
#ifndef SDPA_OCL_NO_GETENV  // TEMP(ttft32), see sdpa_ocl_utils.hpp
        if (const char* brk = std::getenv("OV_GPU_PA_K_TM_BREAK"))
            key_token_major = key_token_major && std::strcmp(brk, "reorder") != 0;
#endif
        jit.make("IS_KEY_TOKEN_MAJOR", key_token_major ? 1 : 0);

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

            const auto max_wg_size = params.get_device_info().max_work_group_size;
            size_t max_heads_per_wg = static_cast<size_t>(max_wg_size / subgroup_size);
            if (max_heads_per_wg == 0) {
                max_heads_per_wg = 1;
            }
            auto wg_heads_number = std::min(heads_number, max_heads_per_wg);
            wgs.global = {sequences_number, heads_number, subgroup_size};
            wgs.local = {1, is_kv_compressed ? 1 : wg_heads_number, subgroup_size};
        }};
    }

private:
    bool m_key_token_major;
};

class PaKVReorderImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PaKVReorderImpl)

    Stage::Ptr kv_reorder = make_stage<PaKVReorderGenerator>(false);
    Stage::Ptr kv_reorder_tm = make_stage<PaKVReorderGenerator>(true);

    PaKVReorderImpl() : PrimitiveImplOCL(PA_KV_reorder::get_type_info_static()) {}
    PaKVReorderImpl(const program_node& node, const RuntimeParams& params) : PaKVReorderImpl() {
        // A static cache shape tells the layout now. A dynamic one is only known at dispatch, so compile the
        // token-major variant too wherever such a page can exist.
        if (!params.is_dynamic()) {
            add_stage(key_cache_token_major(params) ? kv_reorder_tm : kv_reorder, params);
            return;
        }
        add_stage(kv_reorder, params);
        if (token_major_page_possible(params)) {
            add_stage(kv_reorder_tm, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PaKVReorderImpl>(this);
    }

    // Every dispatch resolves the variant from the actual cache shape, so a page this impl has no kernel for (a dynamic
    // impl built where token-major pages cannot be made, given one anyway) throws instead of being misread.
    std::vector<size_t> get_stages_execution_order(const cldnn::kernel_impl_params& params) const override {
        return {stage_for(params)};
    }

    void update_rt_params(const cldnn::primitive_inst& instance) override {
        PrimitiveImplOCL::update_rt_params(instance);
        // Kernel arguments and dispatch data are only set when a stage runs, and the flags above only say what changed
        // since the last run, so a variant that runs for the first time (or again after the other one) gets both.
        const size_t stage = stage_for(*instance.get_impl_params());
        if (stage != m_last_stage) {
            _stages[stage]->kd.need_args_update = true;
            _stages[stage]->kd.need_dispatch_data_update = true;
            m_last_stage = stage;
        }
    }

private:
    size_t stage_for(const cldnn::kernel_impl_params& params) const {
        const bool token_major = key_cache_token_major(params);
        const Stage* wanted = token_major ? kv_reorder_tm.get() : kv_reorder.get();
        for (const auto i : _order) {
            if (_stages[i] == wanted) {
                return i;
            }
        }
        OPENVINO_THROW("[GPU] pa_kv_reorder: no ", token_major ? "token-major" : "d-major", " kernel was compiled for this K cache");
    }

    size_t m_last_stage = std::numeric_limits<size_t>::max();
};

}  // namespace

std::unique_ptr<primitive_impl> PA_KV_reorder::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<cldnn::pa_kv_reorder>());
    return std::make_unique<PaKVReorderImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::pa_kv_reorder)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PaKVReorderImpl)
