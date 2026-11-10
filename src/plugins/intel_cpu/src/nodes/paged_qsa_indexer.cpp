// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/paged_qsa_indexer_kernel.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "cpu_types.h"
#    include "nodes/kernels/sparse_attn/qsa_brgemm.hpp"
#endif

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
using OutputIdx = qsa::paged::OutputIdx;

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
    // Every floating-point data port is independently echoed at whatever precision (f32/f16/bf16) was negotiated
    // upstream -- the kernel reads and writes all of them through row conversions rather than hardcoding f32, since
    // index_key_cache/index_block_rope_cache/block_index_keys_cache grow every step and forcing a whole-tensor Reorder
    // on every call would redo that conversion for the entire history each time. The page-table and index inputs stay
    // i32 -- reordering them to f32 would numerically convert the index values instead of reinterpreting them. The two
    // outputs are always i32.
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
    m_impl = qsa::kernel_impl_type();
#ifdef OPENVINO_ARCH_X86_64
    const auto summary_precision = getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE)->getDesc().getPrecision();
    // The scoring on brgemm where the CPU, the precisions and the head counts allow it; the query and key rows are
    // [tokens, heads * dim], the token count is dynamic. The query is rounded to the precision of the summary blocks
    // for the brgemm products: only done when it is that precision already.
    const auto& c = m_config;
    const auto query_precision = getSrcMemoryAtPort(InputIdx::INDEX_QUERY)->getDesc().getPrecision();
    const auto q_width = getInputShapeAtPort(InputIdx::INDEX_QUERY).getDims()[1];
    const auto k_width = getInputShapeAtPort(InputIdx::RAW_INDEX_KEYS).getDims()[1];
    if (q_width != Shape::UNDEFINED_DIM && k_width != Shape::UNDEFINED_DIM && c.indexer_head_dim != 0 &&
        query_precision == summary_precision) {
        const size_t q_heads = q_width / c.indexer_head_dim;
        const size_t k_heads = k_width / c.indexer_head_dim;
        if (k_heads != 0 && q_heads % k_heads == 0) {
            m_brgemm = qsa::QsaBrgemm::get(
                context->getParamsCache(),
                qsa::QsaBrgemm::Config{q_heads,
                                       k_heads,
                                       c.num_selection_heads,
                                       c.indexer_head_dim,
                                       c.indexer_rotary_dim,
                                       c.block_topk,
                                       static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
                                       c.epsilon,
                                       ov::op::internal::PagedQSAIndexer::effective_scale(c),
                                       summary_precision});
        }
    }
    if (m_brgemm) {
        m_impl = qsa::score_impl_type(summary_precision);
    }
#endif
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_impl);
}

void PagedQSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    std::vector<MemoryPtr> outputs(OutputIdx::COUNT);
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = getDstMemoryAtPort(i);
    }
    const qsa::paged::PagedPorts ports(inputs, outputs);
    const auto cpu_parallel = context->getCpuParallel();
    const auto stats = qsa::paged::seq_stats(ports, m_config);

    // The calls that have complete blocks to score go to the brgemm scorer, which plans its scratch from the sizes of
    // the call; the other calls are scored by the kernel itself.
    const qsa::QsaBrgemm* brgemm = nullptr;
    size_t scorer_shared_bytes = 0;
    size_t scorer_thread_bytes = 0;
#ifdef OPENVINO_ARCH_X86_64
    if (m_brgemm && stats.max_blocks > 0) {
        const auto plan = m_brgemm->plan(qsa::paged::num_sequences(ports), stats.max_q_len, stats.max_blocks);
        brgemm = m_brgemm.get();
        scorer_shared_bytes = plan.shared_bytes;
        scorer_thread_bytes = plan.thread_bytes;
    }
#endif
    const qsa::paged::PagedScratchLayout layout(ports,
                                                m_config,
                                                stats.max_blocks,
                                                scorer_shared_bytes,
                                                scorer_thread_bytes,
                                                static_cast<size_t>(cpu_parallel->get_num_worker_threads()));
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{layout.total_bytes}));

    ov::Extensions::Cpu::XARCH::paged_qsa_indexer_compute(ports,
                                                          m_config,
                                                          layout,
                                                          scratch->getDataAs<uint8_t>(),
                                                          brgemm,
                                                          cpu_parallel);
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
