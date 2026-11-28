// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/paged_page_table_check.hpp"
#include "nodes/kernels/sparse_attn/paged_qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor_cache.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;

// The outputs of the op.
struct OutputIdx {
    static constexpr size_t SEL_INDICES = 0;
    static constexpr size_t SEL_COUNT = 1;
    static constexpr size_t COUNT = 2;
};

}  // namespace

PagedQSAIndexer::PagedQSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto paged_qsa_indexer = ov::as_type_ptr<const ov::op::internal::PagedQSAIndexer>(op);
    CPU_NODE_ASSERT(paged_qsa_indexer, "Expected PagedQSAIndexer node");
    m_config = paged_qsa_indexer->get_config();
    // Fixed by the model (isSupportedOperation checked they are static).
    m_index_query_heads = paged_qsa_indexer->get_index_query_heads();
    m_index_key_heads = paged_qsa_indexer->get_index_key_heads();
}

void PagedQSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every floating-point port is echoed at the precision negotiated upstream (the op validates it is the same), so
    // the growing caches need no Reorder: the kernel converts rows itself. The page-table and index inputs stay i32;
    // the outputs are i32.
    std::vector<PortConfigurator> in_port_configs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_i32_port = i == InputIdx::PAST_LENS || i == InputIdx::SUBSEQUENCE_BEGINS ||
                                 i == InputIdx::BLOCK_INDICES || i == InputIdx::BLOCK_INDICES_BEGINS;
        const ov::element::Type precision = is_i32_port ? ov::element::i32 : getOriginalInputPrecisionAtPort(i);
        in_port_configs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> out_port_configs;
    for (size_t i = 0; i < OutputIdx::COUNT; ++i) {
        out_port_configs.emplace_back(LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(i), false, -1);
    }
    addSupportedPrimDesc(in_port_configs, out_port_configs, impl_desc_type::ref_any);
}

void PagedQSAIndexer::createPrimitive() {
    // The summary blocks are in the precision the block scoring works in.
    const auto precision = getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE)->getDesc().getPrecision();
    const auto& c = m_config;
    ov::Extensions::Cpu::qsa::Config config;
    config.key_block_size = c.key_block_size;
    config.num_selection_heads = c.num_selection_heads;
    config.block_topk = c.block_topk;
    config.head_dim = c.indexer_head_dim;
    config.rotary_dim = c.indexer_rotary_dim;
    config.index_query_heads = m_index_query_heads;
    config.index_key_heads = m_index_key_heads;
    config.num_threads = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
    config.eps = c.epsilon;
    config.scale = ov::op::internal::PagedQSAIndexer::effective_scale(c);
    config.precision = precision;
    config.query_norm = c.query_norm;
    config.query_rope = c.query_rope;
    // The executor is shared by the layers of the same configuration.
    const auto cache = context->getParamsCache();
    [[maybe_unused]] const auto [paged_qsa_indexer_executor, lookup_status] =
        cache->getOrCreate(ov::Extensions::Cpu::qsa::QsaExecutorKey{config},
                           [&cache](const ov::Extensions::Cpu::qsa::QsaExecutorKey& key) {
                               return ov::Extensions::Cpu::XARCH::make_paged_indexer_executor(key.config, cache);
                           });
    CPU_NODE_ASSERT(paged_qsa_indexer_executor, "executor creation fails");
    m_executor = paged_qsa_indexer_executor;
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->impl_type());
}

void PagedQSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    ov::Extensions::Cpu::qsa::PagedIndexerArgs args;
    args.q_idx.reset(getSrcMemoryAtPort(InputIdx::INDEX_QUERY));
    args.k_idx.reset(getSrcMemoryAtPort(InputIdx::RAW_INDEX_KEYS));
    args.rope_cos.reset(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    args.rope_sin.reset(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    args.q_norm_weight.reset(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    args.k_norm_weight.reset(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    args.past_lens.reset(getSrcMemoryAtPort(InputIdx::PAST_LENS));
    args.subsequence_begins.reset(getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    args.block_indices.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES));
    args.block_indices_begins.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES_BEGINS));
    args.raw_k_cache.reset(getSrcMemoryAtPort(InputIdx::INDEX_KEY_CACHE));
    args.indexer_rope_cache.reset(getSrcMemoryAtPort(InputIdx::INDEX_BLOCK_ROPE_CACHE));
    args.summary_cache.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    args.sel_indices.reset(getDstMemoryAtPort(OutputIdx::SEL_INDICES));
    args.sel_count.reset(getDstMemoryAtPort(OutputIdx::SEL_COUNT));

    // Core shape inference leaves the cache layout to the backend: the caches must match the config, or the kernels
    // would read / write out of bounds.
    const auto& c = m_config;
    // The page size is the one of the cache, which the model may leave dynamic.
    const size_t page_size = args.raw_k_cache.size(2);
    CPU_NODE_ASSERT(page_size > 0 && page_size % c.key_block_size == 0,
                    "the cache page size ",
                    page_size,
                    " must be a positive multiple of key_block_size ",
                    c.key_block_size);
    const size_t slots_per_page = page_size / c.key_block_size;
    args.raw_k_cache.assert_dims({0, m_index_key_heads, page_size, c.indexer_head_dim}, true);
    args.indexer_rope_cache.assert_dims({0, slots_per_page, 2 * c.indexer_rotary_dim}, true);
    args.summary_cache.assert_dims({0, m_index_key_heads, slots_per_page, c.indexer_head_dim}, true);
    // The page table is validated once per call, so that the kernels index the caches without bounds checks.
    ov::intel_cpu::check_paged_page_table(
        args.past_lens,
        args.subsequence_begins,
        args.block_indices,
        args.block_indices_begins,
        args.q_idx.size(0),
        page_size,
        std::min({args.raw_k_cache.size(0), args.indexer_rope_cache.size(0), args.summary_cache.size(0)}));

    m_executor->find_batch_max_params(args);
    // One scratch for all the stages: a second createScratchPadMem() could free the memory of the first.
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{m_executor->scratch_bytes(args)}));
    m_executor->execute(args, scratch->getDataAs<uint8_t>(), context->getCpuParallel());
}

bool PagedQSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto paged_qsa_indexer = ov::as_type_ptr<const ov::op::internal::PagedQSAIndexer>(op);
        if (!paged_qsa_indexer) {
            errorMessage = "Only PagedQSAIndexer operation is supported";
            return false;
        }
        // The head counts are fixed once, when the executor is built; the tokens, the number of pages and the page size
        // may change.
        const auto& query_ps = op->get_input_partial_shape(InputIdx::INDEX_QUERY);
        const auto& key_ps = op->get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
        const auto& cache_ps = op->get_input_partial_shape(InputIdx::INDEX_KEY_CACHE);
        if (query_ps.rank().is_dynamic() || query_ps.size() != 2 || query_ps[1].is_dynamic() ||
            key_ps.rank().is_dynamic() || key_ps.size() != 2 || key_ps[1].is_dynamic() ||
            cache_ps.rank().is_dynamic() || cache_ps.size() != 4) {
            errorMessage = "PagedQSAIndexer CPU implementation requires static head counts and a rank-4 key cache.";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::INDEX_QUERY,
                                                 InputIdx::RAW_INDEX_KEYS,
                                                 InputIdx::ROPE_COS,
                                                 InputIdx::ROPE_SIN,
                                                 InputIdx::Q_NORM_WEIGHT,
                                                 InputIdx::K_NORM_WEIGHT,
                                                 InputIdx::INDEX_KEY_CACHE,
                                                 InputIdx::INDEX_BLOCK_ROPE_CACHE,
                                                 InputIdx::BLOCK_INDEX_KEYS_CACHE};
        for (auto port : float_ports) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "PagedQSAIndexer CPU implementation only supports f32, f16, or bf16 data inputs.";
                return false;
            }
        }
        static constexpr size_t i32_ports[] = {InputIdx::PAST_LENS,
                                               InputIdx::SUBSEQUENCE_BEGINS,
                                               InputIdx::BLOCK_INDICES,
                                               InputIdx::BLOCK_INDICES_BEGINS};
        for (auto port : i32_ports) {
            if (op->get_input_element_type(port) != ov::element::i32) {
                errorMessage = "PagedQSAIndexer CPU implementation requires i32 index inputs.";
                return false;
            }
        }
        std::string reason;
        const auto& config = paged_qsa_indexer->get_config();
        if (!ov::op::internal::PagedQSAIndexer::validate_config(config, reason)) {
            errorMessage = "PagedQSAIndexer configuration is invalid: " + reason;
            return false;
        }
        // A static page size is checked here, a dynamic one when the node runs.
        if (cache_ps[2].is_static()) {
            const auto page_size = static_cast<size_t>(cache_ps[2].get_length());
            if (page_size == 0 || page_size % config.key_block_size != 0) {
                errorMessage = "PagedQSAIndexer page size must be a positive multiple of key_block_size.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
