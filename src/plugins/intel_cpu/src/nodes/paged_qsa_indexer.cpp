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

#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/paged_qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/paged_page_table_check.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "cpu_types.h"
#    include "nodes/kernels/sparse_attn/qsa_brgemm.hpp"
#    include "nodes/kernels/sparse_attn/qsa_brgemm_kernel.hpp"
#endif

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;

// Fixed output order of ov::op::internal::PagedQSAIndexer (not part of the op's public API).
struct OutputIdx {
    static constexpr size_t SEL_INDICES = 0;  // [tokens, h_sel, block_topk]
    static constexpr size_t SEL_COUNT = 1;    // [tokens, h_sel]

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
    OPENVINO_ASSERT(paged_qsa_indexer, "Expected PagedQSAIndexer node");
    m_config = paged_qsa_indexer->get_config();
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
    const auto summary_precision = getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE)->getDesc().getPrecision();
    const auto& c = m_config;
    const ov::Extensions::Cpu::qsa::Config config{
        c.key_block_size,
        c.num_selection_heads,
        c.block_topk,
        c.indexer_head_dim,
        c.indexer_rotary_dim,
        static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
        c.epsilon,
        ov::op::internal::PagedQSAIndexer::effective_scale(c),
        summary_precision,
        false};
    m_executor = ov::Extensions::Cpu::XARCH::make_paged_qsa_executor(config);
    m_impl = qsa_kernel_impl_type();
#ifdef OPENVINO_ARCH_X86_64
    // Scoring on brgemm where the CPU, the precisions and the head counts allow it. Query and key rows are
    // [tokens, heads * dim] with a dynamic token count. The brgemm products need the query in the summary precision,
    // so it is used only when it already has it.
    const auto query_precision = getSrcMemoryAtPort(InputIdx::INDEX_QUERY)->getDesc().getPrecision();
    const auto q_width = getInputShapeAtPort(InputIdx::INDEX_QUERY).getDims()[1];
    const auto k_width = getInputShapeAtPort(InputIdx::RAW_INDEX_KEYS).getDims()[1];
    if (q_width != Shape::UNDEFINED_DIM && k_width != Shape::UNDEFINED_DIM && c.indexer_head_dim != 0 &&
        query_precision == summary_precision) {
        const size_t q_heads = q_width / c.indexer_head_dim;
        const size_t k_heads = k_width / c.indexer_head_dim;
        if (k_heads != 0 && q_heads % k_heads == 0) {
            const auto brgemm = ov::Extensions::Cpu::qsa::QsaBrgemm::get(
                context->getParamsCache(),
                ov::Extensions::Cpu::qsa::QsaBrgemm::Params{q_heads,
                                                            k_heads,
                                                            c.num_selection_heads,
                                                            c.indexer_head_dim,
                                                            c.indexer_rotary_dim,
                                                            c.block_topk,
                                                            config.num_threads,
                                                            c.epsilon,
                                                            config.scale,
                                                            summary_precision});
            if (brgemm) {
                m_scorer = ov::Extensions::Cpu::XARCH::make_paged_qsa_scorer(brgemm, config);
                m_impl = ov::Extensions::Cpu::qsa::QsaBrgemm::impl_type(summary_precision);
            }
        }
    }
#endif
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_impl);
}

void PagedQSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    ov::Extensions::Cpu::qsa::PagedCall call;
    call.q_idx.reset(getSrcMemoryAtPort(InputIdx::INDEX_QUERY));
    call.k_idx.reset(getSrcMemoryAtPort(InputIdx::RAW_INDEX_KEYS));
    call.rope_cos.reset(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    call.rope_sin.reset(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    call.q_norm_weight.reset(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    call.k_norm_weight.reset(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    call.past_lens.reset(getSrcMemoryAtPort(InputIdx::PAST_LENS));
    call.subsequence_begins.reset(getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    call.block_indices.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES));
    call.block_indices_begins.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES_BEGINS));
    call.raw_k_cache.reset(getSrcMemoryAtPort(InputIdx::INDEX_KEY_CACHE));
    call.indexer_rope_cache.reset(getSrcMemoryAtPort(InputIdx::INDEX_BLOCK_ROPE_CACHE));
    call.summary_cache.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    call.sel_indices.reset(getDstMemoryAtPort(OutputIdx::SEL_INDICES));
    call.sel_count.reset(getDstMemoryAtPort(OutputIdx::SEL_COUNT));

    check_paged_page_table(call.past_lens,
                           call.subsequence_begins,
                           call.block_indices,
                           call.block_indices_begins,
                           call.q_idx.size(0),
                           call.raw_k_cache.size(2),
                           std::min(call.raw_k_cache.size(0), call.summary_cache.size(0)));

    // The calls that have complete blocks to score go to the brgemm scorer, the other calls are scored by the executor.
    const bool brgemm = m_scorer && m_scorer->applies(call);
    size_t scratch_bytes = m_executor->scratch_bytes(call);
    if (brgemm) {
        scratch_bytes = std::max(scratch_bytes, m_scorer->scratch_bytes(call));
    }
    // One scratch for both stages: a second createScratchPadMem() could free the memory of the first.
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{scratch_bytes}));
    auto* const memory = scratch->getDataAs<uint8_t>();

    const auto cpu_parallel = context->getCpuParallel();
    m_executor->prepare(call, memory, cpu_parallel);
    if (brgemm) {
        m_scorer->score(call, memory, cpu_parallel);
    } else {
        m_executor->score_tokens(call, memory, cpu_parallel);
    }
}

bool PagedQSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        const auto paged_qsa_indexer = ov::as_type_ptr<const ov::op::internal::PagedQSAIndexer>(op);
        if (!paged_qsa_indexer) {
            errorMessage = "Only PagedQSAIndexer operation is supported";
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
        if (!ov::op::internal::PagedQSAIndexer::validate_config(paged_qsa_indexer->get_config(), reason)) {
            errorMessage = "PagedQSAIndexer configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
