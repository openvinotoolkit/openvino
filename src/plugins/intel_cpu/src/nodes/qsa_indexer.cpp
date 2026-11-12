// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_common.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm.hpp"
#    include "nodes/kernels/sparse_attn/qsa_brgemm_kernel.hpp"
#endif

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::intel_cpu::QSAIndexerNode::InputIdx;
using OutputIdx = ov::intel_cpu::QSAIndexerNode::OutputIdx;

// Output shapes of QSAIndexer: everything follows the input shapes, but the summary_cache block count of the longest
// row, which is read from the key_length values.
class QSAIndexerShapeInfer : public ShapeInferEmptyPads {
public:
    explicit QSAIndexerShapeInfer(const ov::intel_cpu::QSAIndexerNode::Config& config) : m_config(config) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& q_dims = input_shapes[InputIdx::Q_IDX].get();
        const auto& summary_dims = input_shapes[InputIdx::SUMMARY_CACHE].get();
        const size_t batch = q_dims[0];
        const auto& key_length = data_dependency.at(InputIdx::KEY_LENGTH);
        const auto* key_length_p = key_length->getDataAs<const int32_t>();
        // key_length is [batch] or [1] (the same length for all the rows)
        const size_t key_length_count = input_shapes[InputIdx::KEY_LENGTH].get()[0];
        int64_t longest = 0;
        for (size_t b = 0; b < key_length_count; b++) {
            longest = std::max<int64_t>(longest, key_length_p[b]);
        }
        const auto num_blocks = static_cast<size_t>(longest / static_cast<int64_t>(m_config.compress_ratio));
        return {{{batch, m_config.h_sel, q_dims[1], m_config.block_topk},
                 {batch, m_config.h_sel, q_dims[1]},
                 {batch, summary_dims[1], num_blocks, summary_dims[3]},
                 input_shapes[InputIdx::PAST_INCOMPLETE_RAW_K].get(),
                 input_shapes[InputIdx::PAST_INCOMPLETE_ROPE].get()},
                ShapeInferStatus::success};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return 1U << InputIdx::KEY_LENGTH;
    }

private:
    ov::intel_cpu::QSAIndexerNode::Config m_config;
};

class QSAIndexerShapeInferFactory : public ShapeInferFactory {
public:
    explicit QSAIndexerShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        if (const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexerNode>(op)) {
            m_config = qsa_indexer->get_config();
        }
    }

    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<QSAIndexerShapeInfer>(m_config);
    }

private:
    ov::intel_cpu::QSAIndexerNode::Config m_config;
};

}  // namespace

QSAIndexer::QSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, QSAIndexerShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexerNode>(op);
    OPENVINO_ASSERT(qsa_indexer, "Expected QSAIndexer node");
    m_config = qsa_indexer->get_config();
}

void QSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every floating-point port (data inputs, state inputs and state outputs) is declared with the
    // precision of Q_IDX, so the kernel sees a single precision (a mismatching edge gets a Reorder).
    // KEY_LENGTH stays i32 -- reordering it to f32 would numerically convert the index values.
    // sel_indices/sel_count are i32; the state outputs never share memory with their inputs: the update is
    // out-of-place.
    const auto data_precision = getOriginalInputPrecisionAtPort(InputIdx::Q_IDX);

    std::vector<PortConfigurator> in_port_configs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const ov::element::Type precision = i == InputIdx::KEY_LENGTH ? ov::element::i32 : data_precision;
        in_port_configs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> out_port_configs;
    for (size_t i = 0; i < OutputIdx::COUNT; ++i) {
        const bool is_index = i == OutputIdx::SEL_INDICES || i == OutputIdx::SEL_COUNT;
        out_port_configs.emplace_back(LayoutType::ncsp,
                                      is_index ? ov::element::i32 : data_precision,
                                      getOutputShapeAtPort(i),
                                      false,
                                      -1);
    }
    addSupportedPrimDesc(in_port_configs, out_port_configs, impl_desc_type::ref_any);
}

void QSAIndexer::createPrimitive() {
    const auto precision = getSrcMemoryAtPort(InputIdx::Q_IDX)->getDesc().getPrecision();
    const auto& c = m_config;
    const ov::Extensions::Cpu::qsa::Config config{
        c.compress_ratio,
        c.h_sel,
        c.block_topk,
        c.indexer_head_dim,
        c.indexer_rotary_dim,
        static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
        c.eps,
        QSAIndexerNode::effective_scale(c),
        precision,
        c.order_by_score};
    m_executor = ov::Extensions::Cpu::XARCH::make_dense_qsa_executor(config);
    m_impl = qsa_kernel_impl_type();
#ifdef OPENVINO_ARCH_X86_64
    // The scoring on brgemm where the CPU, the precision and the head counts allow it; the head counts are static, the
    // batch and the window are not. The scorer is shared by the layers of the same configuration.
    const size_t q_heads = getInputShapeAtPort(InputIdx::Q_IDX).getDims()[2];
    const size_t k_heads = getInputShapeAtPort(InputIdx::K_IDX).getDims()[2];
    if (q_heads != Shape::UNDEFINED_DIM && k_heads != Shape::UNDEFINED_DIM) {
        const auto brgemm =
            ov::Extensions::Cpu::qsa::QsaBrgemm::get(context->getParamsCache(),
                                                     ov::Extensions::Cpu::qsa::QsaBrgemm::Params{q_heads,
                                                                                                 k_heads,
                                                                                                 c.h_sel,
                                                                                                 c.indexer_head_dim,
                                                                                                 c.indexer_rotary_dim,
                                                                                                 c.block_topk,
                                                                                                 config.num_threads,
                                                                                                 c.eps,
                                                                                                 config.scale,
                                                                                                 precision,
                                                                                                 c.order_by_score});
        if (brgemm) {
            m_scorer = ov::Extensions::Cpu::XARCH::make_dense_qsa_scorer(brgemm, config);
            m_impl = ov::Extensions::Cpu::qsa::QsaBrgemm::impl_type(precision);
        }
    }
#endif
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_impl);
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    ov::Extensions::Cpu::qsa::DenseCall call;
    call.q_idx.reset(getSrcMemoryAtPort(InputIdx::Q_IDX));
    call.k_idx.reset(getSrcMemoryAtPort(InputIdx::K_IDX));
    call.rope_cos.reset(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    call.rope_sin.reset(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    call.key_length.reset(getSrcMemoryAtPort(InputIdx::KEY_LENGTH));
    call.q_norm_weight.reset(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    call.k_norm_weight.reset(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    // The state inputs are read-only; the updated state lives in the matching outputs.
    call.past_raw_k_in.reset(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_RAW_K));
    call.past_rope_in.reset(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_ROPE));
    call.summary_cache_in.reset(getSrcMemoryAtPort(InputIdx::SUMMARY_CACHE));
    call.summary_cache.reset(getDstMemoryAtPort(OutputIdx::SUMMARY_CACHE));
    call.past_raw_k.reset(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_RAW_K));
    call.past_rope.reset(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_ROPE));
    call.sel_indices.reset(getDstMemoryAtPort(OutputIdx::SEL_INDICES));
    call.sel_count.reset(getDstMemoryAtPort(OutputIdx::SEL_COUNT));

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

bool QSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto qsa_indexer = ov::as_type_ptr<const ov::intel_cpu::QSAIndexerNode>(op);
        if (!qsa_indexer) {
            errorMessage = "Only QSAIndexer operation is supported";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::Q_IDX,
                                                 InputIdx::K_IDX,
                                                 InputIdx::ROPE_COS,
                                                 InputIdx::ROPE_SIN,
                                                 InputIdx::Q_NORM_WEIGHT,
                                                 InputIdx::K_NORM_WEIGHT,
                                                 InputIdx::SUMMARY_CACHE,
                                                 InputIdx::PAST_INCOMPLETE_RAW_K,
                                                 InputIdx::PAST_INCOMPLETE_ROPE};
        // All floating-point ports share one precision (the op producer sets it).
        const auto data_precision = op->get_input_element_type(InputIdx::Q_IDX);
        if (data_precision != ov::element::f32 && data_precision != ov::element::f16 &&
            data_precision != ov::element::bf16) {
            errorMessage = "QSAIndexer CPU implementation only supports f32, f16, or bf16 data inputs.";
            return false;
        }
        for (auto port : float_ports) {
            if (op->get_input_element_type(port) != data_precision) {
                errorMessage = "QSAIndexer CPU implementation requires all floating-point inputs to have the same "
                               "precision.";
                return false;
            }
        }
        static constexpr size_t i32_ports[] = {InputIdx::KEY_LENGTH};
        for (auto port : i32_ports) {
            if (op->get_input_element_type(port) != ov::element::i32) {
                errorMessage = "QSAIndexer CPU implementation requires i32 index inputs.";
                return false;
            }
        }
        std::string reason;
        if (!ov::intel_cpu::QSAIndexerNode::validate_config(qsa_indexer->get_config(), reason)) {
            errorMessage = "QSAIndexer configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
