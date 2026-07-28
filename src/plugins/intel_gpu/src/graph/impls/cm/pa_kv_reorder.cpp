// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pa_kv_reorder.hpp"

#include <memory>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "paged_attention_gen.hpp"
#include "primitive_cm_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;

namespace {

constexpr size_t reorder_sub_block_size = 16;

// All geometry comes from the primitive descriptor (filled by plugin/ops/pa_kv_reorder.cpp
// reading the static dims of the KV cache Parameter). This avoids reading layout shapes in
// get_jit_constants(), which is invoked at compile time when the runtime layout may still
// be dynamic.
struct KVCacheGeometry {
    size_t block_size = 0;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
};

KVCacheGeometry resolve_geometry_from_desc(const std::shared_ptr<const pa_kv_reorder>& desc) {
    OPENVINO_ASSERT(desc->kv_heads_num > 0, "[GPU][CM] pa_kv_reorder expects positive kv_heads_num");
    OPENVINO_ASSERT(desc->adjusted_k_head_size > 0 && desc->adjusted_v_head_size > 0,
                    "[GPU][CM] pa_kv_reorder expects positive adjusted head sizes");
    OPENVINO_ASSERT(desc->adjusted_paged_attention_block_size > 0, "[GPU][CM] pa_kv_reorder expects positive block size");

    const bool is_kv_compressed = desc->is_kv_compressed;
    const bool is_key_by_channel = desc->is_key_by_channel;
    const size_t scales_zp = desc->scales_zp_size;

    KVCacheGeometry g;

    // adjusted_paged_attention_block_size already accounts for the per-channel scale/zp tail
    // in mode 2 ("block_size + scales_zp_size"), so subtract scales_zp_size to recover the
    // logical block_size. For modes 0/1 the adjusted value equals the logical value.
    if (is_kv_compressed && is_key_by_channel) {
        g.block_size = desc->adjusted_paged_attention_block_size - scales_zp;
    } else {
        g.block_size = desc->adjusted_paged_attention_block_size;
    }

    // adjusted_k_head_size = head_size + scales_zp_size for per-token int8 K (mode 1);
    // for modes 0 and 2 it equals the logical head_size.
    if (is_kv_compressed && !is_key_by_channel) {
        OPENVINO_ASSERT(desc->adjusted_k_head_size > scales_zp,
                        "[GPU][CM] adjusted_k_head_size (",
                        desc->adjusted_k_head_size,
                        ") must exceed scales_zp_size (",
                        scales_zp,
                        ")");
        g.k_head_size = desc->adjusted_k_head_size - scales_zp;
    } else {
        g.k_head_size = desc->adjusted_k_head_size;
    }

    // V is per-token in modes 1 and 2; adjusted_v_head_size = head_size + scales_zp_size.
    if (is_kv_compressed) {
        OPENVINO_ASSERT(desc->adjusted_v_head_size > scales_zp,
                        "[GPU][CM] adjusted_v_head_size (",
                        desc->adjusted_v_head_size,
                        ") must exceed scales_zp_size (",
                        scales_zp,
                        ")");
        g.v_head_size = desc->adjusted_v_head_size - scales_zp;
    } else {
        g.v_head_size = desc->adjusted_v_head_size;
    }

    return g;
}

class PaKVReorderGenerator : public KernelGenerator {
public:
    PaKVReorderGenerator() : KernelGenerator("pa_kv_cache_reorder_ref", "_cm") {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        const auto desc = params.typed_desc<pa_kv_reorder>();
        OPENVINO_ASSERT(desc->kv_heads_num > 0, "[GPU][CM] pa_kv_reorder expects positive kv_heads_num in descriptor");

        jit.make("KV_HEADS_NUM", desc->kv_heads_num);

        const bool is_kv_compressed = desc->is_kv_compressed;
        const bool is_key_by_channel = desc->is_key_by_channel;
        const int compression_mode = !is_kv_compressed ? 0 : (is_key_by_channel ? 2 : 1);
        jit.make("KV_CACHE_COMPRESSION", compression_mode);
        jit.make("SUB_BLOCK_SIZE", reorder_sub_block_size);

        const auto geometry = resolve_geometry_from_desc(desc);
        OPENVINO_ASSERT(geometry.block_size > 0, "[GPU][CM] resolved block_size must be positive");

        jit.make("BLOCK_SIZE", geometry.block_size);
        jit.make("K_HEAD_SIZE", geometry.k_head_size);
        jit.make("V_HEAD_SIZE", geometry.v_head_size);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES_BEGINS});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            OPENVINO_ASSERT(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto desc = params.typed_desc<pa_kv_reorder>();
            OPENVINO_ASSERT(desc->kv_heads_num > 0, "[GPU][CM] pa_kv_reorder expects positive kv_heads_num in descriptor");

            const auto& begins_layout = params.input_layouts[pa_kv_reorder::PaKVReorderInputIdx::BLOCK_UPDATE_INDICES_BEGINS];
            const auto begins_len = static_cast<size_t>(begins_layout.get_partial_shape()[0].get_length());
            OPENVINO_ASSERT(begins_len >= 1, "[GPU][CM] BLOCK_UPDATE_INDICES_BEGINS must have at least 1 element");
            const size_t sequences_number = begins_len - 1;

            // One thread per (subsequence, kv_head); slot pairs are processed sequentially
            // in-kernel because src/dst ranges may overlap.
            wgs.global = {std::max<size_t>(1, sequences_number), desc->kv_heads_num, 1};
            wgs.local = {1, 1, 1};
        }};
    }
};

class PaKVReorderImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PaKVReorderImpl)

    Stage::Ptr kv_reorder = make_stage<PaKVReorderGenerator>();

    PaKVReorderImpl() : PrimitiveImplCM(PaKVReorderImplementationManager::get_type_info_static()) {}
    PaKVReorderImpl(const program_node& /*node*/, const RuntimeParams& params) : PaKVReorderImpl() {
        add_stage(kv_reorder, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PaKVReorderImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PaKVReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<pa_kv_reorder>());
    return std::make_unique<PaKVReorderImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PaKVReorderImpl)
