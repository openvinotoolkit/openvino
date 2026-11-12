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
#include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sparse_attention.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::op::v17::SparseAttention: block-sparse group-query attention over a dense
/// `[B, Hkv, S, D]` K/V cache, with the selection from a producer such as QSAIndexer. Causal: query l is at S - L + l.
/// The SparseAttentionWithKVCache form keeps the cache in VariableStateKVcacheBHLS states, plain or u8 / u4.
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

    // No persistent scratch MemoryPtr member, on purpose: the DnnlScratchPad is one shared, grow-only buffer, so a
    // later createScratchPadMem() that grows it frees the memory of an earlier one. execute() makes exactly one
    // combined request and never holds a pointer across it.

#ifdef OPENVINO_ARCH_X86_64
    // AMX matrix products of the prefill over a pack of query heads (empty when the node is not eligible: no AMX, a non
    // bf16/f16 cache, a non-causal op, dynamic head counts or a pack of one head), see SparseBrgemm.
    ov::Extensions::Cpu::SparseBrgemmKernels m_brgemm;
#endif
};

}  // namespace ov::intel_cpu::node
