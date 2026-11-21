// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "block_sparse_attention.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "cm_db/experimental_kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "impls/cm/primitive_cm_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "experimental/primitives/block_sparse_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;

namespace {

constexpr size_t cm_register_file_size = 256;

struct BlockSparseAttentionGeometry {
    size_t B = 0;
    size_t Lq = 0;
    size_t Lk = 0;
    size_t Hq = 0;
    size_t Hkv = 0;
    size_t head_size = 0;
    bool causal = false;
    size_t block_size_q = 0;
    size_t block_size_kv = 0;
    bool has_scale = false;
};

BlockSparseAttentionGeometry resolve_geometry(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<block_sparse_attention>();

    const auto& q_shape = params.input_layouts[BlockSparseAttentionInputIdx::QUERY].get_partial_shape();
    const auto& k_shape = params.input_layouts[BlockSparseAttentionInputIdx::KEY].get_partial_shape();
    const auto& mask_shape = params.input_layouts[BlockSparseAttentionInputIdx::BLOCK_MASK].get_partial_shape();

    // Only batch/num_heads/num_kv_heads/head_size need to be statically known here: those are the
    // only fields get_jit_constants() reads, and get_jit_constants()/get_arguments_desc() run once,
    // at kernel-build time, against the primitive's *declared* layout -- which may still carry a
    // dynamic token-count axis (FlashVSR's streaming DiT graph varies Lq/Lk per chunk). Lq/Lk are
    // read on a best-effort basis below: get_dispatch_data_func()'s lambdas are the only callers
    // that need them, and those are invoked separately, per real execution, against the fully
    // resolved kernel_impl_params (see PrimitiveImplOCL::execute_stage) -- at which point Lq/Lk are
    // concrete and resolve_geometry() picks up the real values.
    OPENVINO_ASSERT(q_shape.rank().is_static() && q_shape[0].is_static() && q_shape[2].is_static() && q_shape[3].is_static(),
                    "[GPU][CM] block_sparse_attention requires a static batch/num_heads/head_size on query");
    OPENVINO_ASSERT(k_shape.rank().is_static() && k_shape[2].is_static(), "[GPU][CM] block_sparse_attention requires a static num_kv_heads on key");

    BlockSparseAttentionGeometry g;
    g.B = static_cast<size_t>(q_shape[0].get_length());
    g.Lq = q_shape[1].is_static() ? static_cast<size_t>(q_shape[1].get_length()) : 0;
    g.Hq = static_cast<size_t>(q_shape[2].get_length());
    g.head_size = static_cast<size_t>(q_shape[3].get_length());
    g.Lk = k_shape[1].is_static() ? static_cast<size_t>(k_shape[1].get_length()) : 0;
    g.Hkv = static_cast<size_t>(k_shape[2].get_length());
    g.causal = desc->causal;
    g.block_size_q = static_cast<size_t>(desc->block_size_q);
    g.block_size_kv = static_cast<size_t>(desc->block_size_kv);
    g.has_scale = params.input_layouts.size() == 5;

    // block_sparse_attention_prefill.cm's SPARSE_BLOCK_SIZE is a single scalar shared by both the
    // Q- and KV-block granularity, and must be 128 or 256 (cm_sdpa_common.hpp's
    // sdpa_kernel_lsc_prefetch static_asserts on exactly this). validate_impl already rejects this
    // case up front when block_size_q/kv are statically known at impl-selection time; this assert
    // is the hard backstop so an unsupported configuration fails loudly here instead of silently
    // misdispatching if it's ever reached (e.g. impl re-selection edge cases).
    OPENVINO_ASSERT(g.block_size_q == g.block_size_kv,
                    "[GPU][CM] block_sparse_attention: block_size_q (",
                    g.block_size_q,
                    ") must equal block_size_kv (",
                    g.block_size_kv,
                    ") -- the ported CM kernel uses a single SPARSE_BLOCK_SIZE for both.");
    OPENVINO_ASSERT(g.block_size_q == 128 || g.block_size_q == 256,
                    "[GPU][CM] block_sparse_attention: block_size_q/block_size_kv must be 128 or 256, got ",
                    g.block_size_q);

    // v1 limitation: the ported kernel's sparse_block_mask indexing assumes a dense
    // [B,Hq,num_q_blocks,num_k_blocks] mask -- no Bm==1/Hm==1|Hkv broadcast support (the core
    // op/CPU reference still allow it). Fail loudly rather than silently reading the wrong mask
    // row if this is ever reached with a broadcast mask. Only mask_shape[0]/[1] (B/Hq) are checked
    // here -- guarded on being statically known, mirroring validate_impl()'s lenient pattern --
    // since mask_shape[2]/[3] (num_q_blocks/num_k_blocks) may be dynamic (FlashVSR's streaming
    // graph) and were never read by this check to begin with.
    OPENVINO_ASSERT(mask_shape.rank().is_static(), "[GPU][CM] block_sparse_attention requires a static-rank block_mask");
    OPENVINO_ASSERT(!mask_shape[0].is_static() || mask_shape[0].get_length() == static_cast<int64_t>(g.B),
                    "[GPU][CM] block_sparse_attention: block_mask broadcasting (Bm=",
                    mask_shape[0],
                    ") is not supported by this implementation -- block_mask must be exactly [B=",
                    g.B,
                    ", Hq=",
                    g.Hq,
                    ", num_q_blocks, num_k_blocks].");
    OPENVINO_ASSERT(!mask_shape[1].is_static() || mask_shape[1].get_length() == static_cast<int64_t>(g.Hq),
                    "[GPU][CM] block_sparse_attention: block_mask broadcasting (Hm=",
                    mask_shape[1],
                    ") is not supported by this implementation -- block_mask must be exactly [B=",
                    g.B,
                    ", Hq=",
                    g.Hq,
                    ", num_q_blocks, num_k_blocks].");

    return g;
}

// q_step (REG_N in the CM kernels' terms): 16 on Xe2, 8 on Xe1. Mirrors VLSDPAGenerator's
// identical derivation from CM_GRF_WIDTH.
size_t get_q_step(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    const size_t cm_grf_width = (info.arch <= gpu_arch::xe_hpc) ? 256 : 512;
    return cm_grf_width / 32;
}

void push_int32_scalar(KernelData& kd, int32_t value) {
    scalar_desc desc;
    desc.t = scalar_desc::Types::INT32;
    desc.v.s32 = value;
    kd.params.scalars.push_back(desc);
}

void push_float_scalar(KernelData& kd, float value) {
    scalar_desc desc;
    desc.t = scalar_desc::Types::FLOAT32;
    desc.v.f32 = value;
    kd.params.scalars.push_back(desc);
}

// Reads the optional runtime `scale` input's scalar value off the device. Only called when
// has_scale is true (5-input primitive variant); scale shares query/key/value's f16 element type
// (enforced by the core op's merged-dtype validation and this impl's f16-only validate_impl).
float read_scale_value(const RuntimeParams& params) {
    const auto scale_mem = params.memory_deps.at(BlockSparseAttentionInputIdx::SCALE);
    mem_lock<ov::float16, mem_lock_type::read> scale_lock(scale_mem, *params.strm);
    return static_cast<float>(scale_lock[0]);
}

// Block-sparse prefill kernel: dense [B,L,H,D] SDPA with SPARSE_BLOCK_SIZE-gated block skipping,
// ported from block_sparse_attention_prefill.cm (itself ported from aboutSHW's
// cm_sdpa_opt_prefill.cm). Used for all shapes (including single-token Lq==1).
class BlockSparseAttentionPrefillGenerator : public ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator {
public:
    BlockSparseAttentionPrefillGenerator() : ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator("block_sparse_attention_prefill") {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=" + std::to_string(cm_register_file_size);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto g = resolve_geometry(params);
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(g.head_size));

        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("CMFLA_NUM_HEADS", g.Hq),
            make_jit_constant("CMFLA_NUM_KV_HEADS", g.Hkv),
            make_jit_constant("CMFLA_HEAD_SIZE", g.head_size),
            make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor),
            make_jit_constant("CMFLA_IS_CAUSAL", g.causal ? 1 : 0),
            // BlockSparseAttention's declared ABI is always heads-inner [B,L,H,D].
            make_jit_constant("Q_TRANSPOSE_ORDER", 1),
            make_jit_constant("K_TRANSPOSE_ORDER", 1),
            make_jit_constant("V_TRANSPOSE_ORDER", 1),
            make_jit_constant("OUTPUT_TRANSPOSE_ORDER", 1),
            make_jit_constant("HAS_ATTN_MASK_INPUT", 0),
            make_jit_constant("HAS_SCALE_INPUT", g.has_scale ? 1 : 0),
            make_jit_constant("SPARSE_BLOCK_SIZE", g.block_size_kv),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        const auto g = resolve_geometry(params);
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, BlockSparseAttentionInputIdx::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, BlockSparseAttentionInputIdx::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, BlockSparseAttentionInputIdx::VALUE});
        uint32_t scalar_idx = 0;
        if (g.has_scale) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, scalar_idx++});  // scale
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, scalar_idx++});  // q_len
        args.push_back({ArgumentDescriptor::Types::SCALAR, scalar_idx++});  // kv_len
        args.push_back({ArgumentDescriptor::Types::INPUT, BlockSparseAttentionInputIdx::BLOCK_MASK});
        args.push_back({ArgumentDescriptor::Types::SCALAR, scalar_idx++});  // num_q_blocks
        args.push_back({ArgumentDescriptor::Types::SCALAR, scalar_idx++});  // num_k_blocks
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            OPENVINO_ASSERT(!params.is_dynamic());
            const auto g = resolve_geometry(params);

            const size_t q_step = get_q_step(params);
            constexpr size_t wg_size = 16;
            const size_t wg_seq_len = wg_size * q_step;
            const size_t wg_count = (g.Lq + wg_seq_len - 1) / wg_seq_len;

            auto& wgs = kd.params.workGroups;
            wgs.global = {1, g.B * g.Hq, wg_count * wg_size};
            wgs.local = {1, 1, wg_size};

            const size_t num_q_blocks = (g.Lq + g.block_size_q - 1) / g.block_size_q;
            const size_t num_k_blocks = (g.Lk + g.block_size_kv - 1) / g.block_size_kv;

            kd.params.scalars.clear();
            if (g.has_scale) {
                push_float_scalar(kd, read_scale_value(params));
            }
            push_int32_scalar(kd, static_cast<int32_t>(g.Lq));
            push_int32_scalar(kd, static_cast<int32_t>(g.Lk));
            push_int32_scalar(kd, static_cast<int32_t>(num_q_blocks));
            push_int32_scalar(kd, static_cast<int32_t>(num_k_blocks));
        }};
    }
};

class BlockSparseAttentionImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::BlockSparseAttentionImpl)

    Stage::Ptr prefill = make_stage<BlockSparseAttentionPrefillGenerator>();

    BlockSparseAttentionImpl() : PrimitiveImplCM(BlockSparseAttentionImplementationManager::get_type_info_static()) {}
    BlockSparseAttentionImpl(const program_node& /*node*/, const RuntimeParams& params) : BlockSparseAttentionImpl() {
        // Single-token (Lq==1) is fully normalized onto the block-sparse prefill kernel; there is
        // no separate decode path anymore. Always register the prefill stage.
        add_stage(prefill, params);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& /*params*/) const override {
        return {};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<BlockSparseAttentionImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> BlockSparseAttentionImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<block_sparse_attention>());
    return std::make_unique<BlockSparseAttentionImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::BlockSparseAttentionImpl)
