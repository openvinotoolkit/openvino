// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_state.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_brgemm.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sparse_attention.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::op::v17::SparseAttention (standard, non-paged,
/// dense-cache "plan consumer" for block-sparse attention).
///
/// Performs sparse, selected-block group-query attention over a dense, already-populated,
/// plain batched `[B, Hkv, S, D]` K/V cache using the clean `(sel_indices, sel_count)` selection
/// contract (block_indices, block_count) emitted by an upstream producer (e.g. QSAIndexer). No RoPE/norm parameters, no
/// projection weights, no output gate, and no in-place KV write. Each query position `l` maps to
/// the bottom-right-aligned logical position `q_abs = S - L + l` for the causal (token-level fine
/// filter) comparison; the producer already applied the block-level coarse filter. QUERY / KEY /
/// VALUE may be negotiated as f32, f16, or bf16 independently. The scalar and legacy AMX-GEMV paths compute in fp32;
/// the brgemm prefill (bf16 / f16 with AMX, causal, more than one query) multiplies bf16 / f16 matrices: the scores and
/// the accumulators are fp32 and the softmax weights are converted to the matrix type before P.V. block_indices /
/// block_count / key_length are always negotiated as i32. v1 scope: the
/// 5-input form (no padding) and the 6-input form (with key_length, honoring per-batch
/// left-padding) are both accepted, with query_block_size == 1; the optional scale and sinks inputs
/// are not supported -- the scale always defaults to `1 / sqrt(k_head_size)`.
///
/// The node also implements ov::intel_cpu::SparseAttentionWithKVCache: the same attention, but the dense K/V cache
/// is owned by a pair of VariableStateKVcacheBHLS states instead of being fed by a Concat. The state is assigned
/// by MemoryInputSparseSDPA, the node appends the current step's key/value into it in place (growing the buffer
/// geometrically, reordering rows by beam_idx when it is not the identity) and runs the very same kernel over the
/// state's BHLS buffer. The cache is kept in the query's precision, or compressed to u8 / u4 with per-token groups of
/// scale / zero point (the key_cache_precision / value_cache_precision hints of the plugin, same as for SDPA); the
/// kernel then dequantizes just the rows it reads, i.e. the selected blocks and the causal tail. The keys may also be
/// quantized by channel (u8, groups of `group_size` tokens, see ov::internal::key_cache_quant_mode). TurboQuant and
/// by-channel u4 are not supported: such a side stays a plain cache in the query's precision.
class SparseSDPA : public Node {
public:
    SparseSDPA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::SparseSDPA;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    // The state inputs of the stateful form are proxies and have zero length on the first step.
    [[nodiscard]] bool isExecutable() const override {
        if (!m_fuse_concat) {
            return Node::isExecutable();
        }
        for (size_t i = 0; i < m_base_inputs + 1; i++) {  // + beam_idx
            if (isInputTensorAtPortEmpty(i)) {
                return false;
            }
        }
        return true;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    // true for SparseAttentionWithKVCache, the form that owns the KV cache states
    [[nodiscard]] bool isStateful() const {
        return m_fuse_concat;
    }

    // KV cache state interface, consumed by MemoryInputSparseSDPA (only meaningful for SparseAttentionWithKVCache).
    void assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx);
    [[nodiscard]] static std::vector<size_t> getKVCacheOrder() {
        return {0, 1, 2, 3};  // BHLS: the kernels read rows along L contiguously
    }
    [[nodiscard]] ov::element::Type getKeyCachePrecision() const {
        return m_key_spec.precision;
    }
    [[nodiscard]] ov::element::Type getValueCachePrecision() const {
        return m_value_spec.precision;
    }
    [[nodiscard]] const ov::Extensions::Cpu::CacheSpec& getKeySpec() const {
        return m_key_spec;
    }
    [[nodiscard]] const ov::Extensions::Cpu::CacheSpec& getValueSpec() const {
        return m_value_spec;
    }
    [[nodiscard]] static bool isQuantizedCache(const ov::Extensions::Cpu::CacheSpec& spec) {
        return spec.precision == ov::element::u8 || spec.precision == ov::element::u4;
    }

private:
    // Appends the current step's key/value to the key/value states.
    void updateKVCache();

    // SparseAttentionWithKVCache: the last three inputs are beam_idx, past_key, past_value.
    bool m_fuse_concat = false;
    // number of the leading (non-stateful) inputs: 5 or 6
    size_t m_base_inputs = 0;
    // precision of the plain (not quantized) cache and the precision the quantized cache is decoded to: the query's
    ov::element::Type m_cache_prec = ov::element::f32;
    // the stored precision of the key / value cache (m_cache_prec, u8 or u4) and its quantization parameters
    ov::Extensions::Cpu::CacheSpec m_key_spec;
    ov::Extensions::Cpu::CacheSpec m_value_spec;
    std::shared_ptr<VariableStateKVcache> m_k_state;
    std::shared_ptr<VariableStateKVcache> m_v_state;

    size_t m_key_block_size = 0;
    bool m_causal = true;
    float m_scale = 0.0F;

    // Note: there is deliberately no persistent MemoryPtr scratch member here. The plugin's
    // DnnlScratchPad is a single shared, grow-only buffer per engine (see dnnl_scratch_pad.h):
    // every createScratchPadMem() call aliases the SAME underlying allocation, and growing it
    // (MemoryBlockWithReuse::resize()) frees the old buffer and mallocs a new one. Holding a
    // pointer from an earlier createScratchPadMem() call across a later one in the same execute()
    // invocation is a use-after-free as soon as the later call grows the pool -- so execute()
    // issues exactly one combined createScratchPadMem() call per invocation (covering both the
    // per-thread f32 scratch and the AMX run-decomposition i64 scratch) and never holds a pointer
    // from a prior call across it.

    // AMX QK^T fast path: ov::element::dynamic means "not eligible" (decided once in
    // createPrimitive(), never re-checked per call). Only activates when QUERY/KEY_CACHE_DENSE
    // are already natively bf16 (or natively f16) end-to-end -- no forced downcast of f32 data.
    ov::element::Type m_amx_prec = ov::element::dynamic;
#ifdef OPENVINO_ARCH_X86_64
    // The JIT kernel requires its block_size (rows per AMX call) to be a multiple of 16, but real
    // sel_block_size values (e.g. 4 for Qwen4) are usually smaller than that -- so AMX calls merge
    // m_amx_unit_blocks adjacent selected blocks (== 16 / gcd(16, sel_block_size)) into one valid
    // 16-row call. m_gemv_qk1 therefore has block_size = m_amx_unit_blocks * sel_block_size, and
    // m_gemv_qk4 has 4x that, to further amortize over longer runs of adjacent selected blocks.
    // Both kernels are safe to share read-only across worker threads: tile_config()/tile_release()
    // only touch per-thread AMX hardware state, and operator() only reads immutable JIT code plus
    // the pointers passed in -- see sparse_sdpa_kernel.cpp's run_amx_qk_chunks() for how they're
    // used together under one tile_config()/tile_release() bracket. A run shorter than
    // m_amx_unit_blocks (or the remainder after merging as many full units as fit) falls back to
    // the scalar per-token path.
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk1;
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk4;
    size_t m_amx_unit_blocks = 0;
    // AMX matrix products of the prefill over a pack of query heads (empty when the node is not eligible: no AMX, a non
    // bf16/f16 cache, a non-causal op, dynamic head counts or a pack of one head), see SparseBrgemm.
    ov::Extensions::Cpu::SparseBrgemmKernels m_brgemm;
#endif
};

}  // namespace ov::intel_cpu::node
