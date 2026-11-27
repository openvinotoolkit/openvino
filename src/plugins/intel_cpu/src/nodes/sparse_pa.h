// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/paged_sparse_attention.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::op::internal::PagedSparseAttention (paged-cache sparse-attention consumer for Qwen
/// Sparse Attention).
///
/// Writes the current-step K/V into the caller-owned paged KV caches, addressed through the shared PagedAttention page
/// table (validated on every call), then runs the same online-softmax attention over the selected blocks and the causal
/// tail as the dense SparseSDPA node, with every K/V read resolved through the page table. The attention is the
/// SparsePAExecutor of the kernels (SIMD, or AMX brgemm for the bf16 / f16 prefill). SCALE is a
/// runtime input; sinks are not supported (SINKS must be an empty tensor). The op has one query per selection row, so
/// the per-sequence selection row offsets are subsequence_begins. Runs of adjacent selected blocks are capped at
/// physical page boundaries, since only blocks of the same page are contiguous in memory.
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
    // Only the token-major inputs gate execution: SINKS is an empty tensor when
    // unused, which must not make the node skip.
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
    /// @brief Kernel parameters: the head counts and sizes come from the runtime info of the op (the cache shapes
    /// are padded when the caches are quantized), the widths and the selection heads from the static input shapes.
    using Config = ov::Extensions::Cpu::SparsePAConfig;
    static bool deriveConfig(const ov::op::internal::PagedSparseAttention& op, Config& config, std::string& reason);
    /// @brief Validates the cache layout against the plugin's KV-cache quantization settings and fills the cache specs
    /// and the page size of m_config. The quantized cache shapes (padded head size or page size) follow PagedAttention,
    /// see CacheSpec.
    bool deriveQuant(std::string& reason);
    [[nodiscard]] bool isQuantized() const {
        return ov::intel_cpu::is_quantized_cache(m_config.key_spec);
    }

    Config m_config{};
    // Shared by the nodes of the same configuration (params cache of the graph context); it owns the AMX kernels of the
    // prefill path and the buffers of the K/V cache write.
    std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> m_executor;
};

}  // namespace ov::intel_cpu::node