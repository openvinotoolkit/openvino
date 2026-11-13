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
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"
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
    /// @brief Writes the current-step K/V into the paged caches (parallel over token x kv head), quantizing them for
    /// u8/u4 caches. The attention kernel only reads the caches, so this is its barrier-separated first phase.
    /// @param scratch at least writeKVScratchBytes() bytes, 64-byte aligned: the slot mapping and the by-channel
    /// requantization buffer live there, not in the node
    void writeKV(uint8_t* scratch);
    /// @brief Bytes of the slot mapping of `tokens` tokens in the scratch of writeKV(), aligned to keep what follows it
    /// aligned.
    [[nodiscard]] static size_t writeKVSlotMappingBytes(size_t tokens);
    /// @brief Bytes of scratch writeKV() needs for `tokens` tokens and caches of `block_size` rows per page.
    [[nodiscard]] size_t writeKVScratchBytes(size_t tokens, size_t block_size) const;
    /// @brief Per-thread scratch layout of the kernel for a selected_block_indices width.
    [[nodiscard]] ov::Extensions::Cpu::SparsePAScratchLayout scratchLayout() const;

    Config m_config{};
    ov::Extensions::Cpu::SparsePAQuant m_quant{};
#ifdef OPENVINO_ARCH_X86_64
    // AMX matrix products of the prefill path (empty when the node is not eligible), see SparseBrgemm.
    ov::Extensions::Cpu::SparseBrgemmKernels m_brgemm;
#endif
    size_t m_max_run_rows = 0;  // most key rows of one run, see SparsePAScratchLayout
};

}  // namespace ov::intel_cpu::node