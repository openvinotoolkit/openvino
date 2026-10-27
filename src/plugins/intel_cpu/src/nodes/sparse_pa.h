// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_pa_kernel.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::op::internal::PagedSparseAttention (paged-cache sparse-
/// attention consumer for Qwen Sparse Attention).
///
/// Writes the current-step K/V into the caller-owned paged main KV cache addressed through
/// the shared PagedAttention page table, then performs the same scalar online-softmax
/// selected-block + causal-tail attention as the dense SparseSDPA node, with every K/V read
/// resolved through that same page table instead of a flat dense `[B, Hkv, S, D]` cache.
/// SCALE is a runtime input; sinks are not supported yet (SINKS must be an empty tensor). The op has one
/// query per selection row, so the per-sequence selection row offsets are subsequence_begins.
/// Selected-block QK^T has a bf16/f16 AMX fast path mirroring SparseSDPA's, adapted for the
/// paged cache: merge runs are additionally capped at physical-page boundaries, since two
/// logically-adjacent selected blocks are only guaranteed memory-contiguous within the same
/// physical page.
class SparsePA : public Node {
public:
    /// @brief Input order of ov::op::internal::PagedSparseAttention.
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [tokens, num_heads*k_head_size]
        static constexpr size_t KEY = 1;                     ///< [tokens, num_kv_heads*k_head_size], current step
        static constexpr size_t VALUE = 2;                   ///< [tokens, num_kv_heads*v_head_size], current step
        static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, num_kv_heads, page_size, k_head_size]
        static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, num_kv_heads, page_size, v_head_size]
        static constexpr size_t PAST_LENS = 5;               ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_sequences+1]
        static constexpr size_t BLOCK_INDICES = 7;           ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_sequences+1]
        static constexpr size_t SELECTED_BLOCK_INDICES = 9;  ///< [tokens, num_selection_heads, K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 10;   ///< [tokens, num_selection_heads]
        static constexpr size_t SCALE = 11;                  ///< scalar
        static constexpr size_t SINKS = 12;                  ///< [num_heads], or an empty tensor (no sinks)
    };

    SparsePA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::SparsePA;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    // Only the token-major inputs gate execution: SINKS (and the optional SELECTED_BLOCK_MASK) is an
    // empty tensor when unused, which must not make the node skip.
    [[nodiscard]] bool neverExecute() const override {
        const auto* pd = getSelectedPrimitiveDescriptor();
        return pd->hasZeroInputDimsAtPort(InputIdx::QUERY) || pd->hasZeroInputDimsAtPort(InputIdx::KEY) ||
               pd->hasZeroInputDimsAtPort(InputIdx::VALUE);
    }
    [[nodiscard]] bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(InputIdx::QUERY) && !isInputTensorAtPortEmpty(InputIdx::KEY) &&
               !isInputTensorAtPortEmpty(InputIdx::VALUE);
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    /// @brief Kernel parameters derived from the static input shapes. The key/value widths are used for the head
    /// sizes rather than the cache shapes, whose last two dims are padded when the caches are quantized.
    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t key_block_size = 0;
        bool causal = true;
    };
    static bool deriveConfig(const ov::op::internal::PagedSparseAttention& op, Config& config, std::string& reason);
    /// @brief Validates the cache layout against the plugin's KV-cache quantization settings and fills m_quant. The
    /// quantized cache shapes (padded head size or page size) follow PagedAttention, see SparsePAQuant.
    bool deriveQuant(std::string& reason);
    /// @brief Quantizes the current-step K/V into u8/u4 caches (the kernel only reads quantized caches).
    void writeQuantizedKV();

    Config m_config{};
    ov::Extensions::Cpu::SparsePAQuant m_quant{};
    ov::intel_cpu::PlainTensor m_slot_mapping;  // [tokens], physical slot of every new token
    ov::intel_cpu::PlainTensor m_temp_buffer;   // by-channel requantization scratch of paged_attn_quantkv

    // AMX QK^T fast path: ov::element::dynamic means "not eligible" (decided once in
    // createPrimitive(), never re-checked per call). Only activates when QUERY/KEY_CACHE are
    // already natively bf16 (or natively f16) end-to-end -- no forced downcast of f32 data. The
    // AMX path reads exclusively from KEY_CACHE/VALUE_CACHE (never the transient current-step
    // KEY/VALUE inputs), so KEY_CACHE's negotiated precision -- not KEY's -- is what must match
    // QUERY and the target ISA.
    ov::element::Type m_amx_prec = ov::element::dynamic;
#ifdef OPENVINO_ARCH_X86_64
    // The JIT kernel requires its block_size (rows per AMX call) to be a multiple of 16, but real
    // key_block_size values (e.g. 4 for Qwen4) are usually smaller than that -- so AMX calls merge
    // m_amx_unit_blocks adjacent selected blocks (== 16 / gcd(16, key_block_size)) into one valid
    // 16-row call. m_gemv_qk1 therefore has block_size = m_amx_unit_blocks * key_block_size, and
    // m_gemv_qk4 has 4x that, to further amortize over longer runs of adjacent selected blocks --
    // never built at all if the physical page (KEY_CACHE's dim 2, static only) can't even fit one full unit.
    // Both kernels are safe to share read-only across worker threads: tile_config()/tile_release()
    // only touch per-thread AMX hardware state, and operator() only reads immutable JIT code plus
    // the pointers passed in. A run shorter than m_amx_unit_blocks (or the remainder after merging
    // as many full units as fit, or one capped at a physical-page boundary) falls back to the
    // scalar per-token path.
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk1;
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk4;
    size_t m_amx_unit_blocks = 0;
#endif
};

}  // namespace ov::intel_cpu::node