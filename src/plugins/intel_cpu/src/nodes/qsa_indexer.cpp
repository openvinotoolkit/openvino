// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/sparse_attn/qsa_brgemm_scorer.hpp"
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

#ifdef OPENVINO_ARCH_X86_64
// Query heads per K head if the scoring of this node (precision of the data ports `precision`) runs on brgemm, 0 if
// not.
static size_t brgemm_group_size(const Node& node,
                                const ov::element::Type& precision,
                                size_t selection_heads,
                                size_t dim) {
    const auto q_heads = node.getInputShapeAtPort(InputIdx::Q_IDX).getDims()[2];
    const auto k_heads = node.getInputShapeAtPort(InputIdx::K_IDX).getDims()[2];
    if (q_heads == Shape::UNDEFINED_DIM || k_heads == Shape::UNDEFINED_DIM) {
        return 0;
    }
    return qsa::brgemm_score_group_size(precision, q_heads, k_heads, selection_heads, dim);
}
#endif

void QSAIndexer::createPrimitive() {
#ifdef OPENVINO_ARCH_X86_64
    // The scorer depends on the configuration, the static head counts and the precision only: like the executor of the
    // ScaledDotProductAttention it comes from the params cache once here, never while the model runs.
    // initSupportedPrimitiveDescriptors() registered the brgemm implementation type for the nodes that can have it.
    const auto precision =
        getSelectedPrimitiveDescriptor()->getConfig().inConfs[InputIdx::Q_IDX].getMemDesc()->getPrecision();
    const auto q_heads = getInputShapeAtPort(InputIdx::Q_IDX).getDims()[2];
    const auto k_heads = getInputShapeAtPort(InputIdx::K_IDX).getDims()[2];
    if (q_heads != Shape::UNDEFINED_DIM && k_heads != Shape::UNDEFINED_DIM) {
        const qsa::BrgemmBlockScorer::Config config{
            q_heads,
            k_heads,
            m_config.h_sel,
            m_config.indexer_head_dim,
            m_config.indexer_rotary_dim,
            m_config.block_topk,
            static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
            m_config.eps,
            ov::intel_cpu::QSAIndexerNode::effective_scale(m_config),
            precision,
            m_config.order_by_score};
        m_scorer = qsa::BrgemmBlockScorer::get(context->getParamsCache(), config);
    }
    if (!m_scorer) {
        // nothing to score with on brgemm: the scalar scoring runs, so the node is a reference implementation
        getSelectedPrimitiveDescriptor()->setImplementationType(impl_desc_type::ref);
    }
#endif
}

void QSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // Every floating-point port (data inputs, state inputs and state outputs) is declared with the
    // precision of Q_IDX, so the kernel sees a single precision (a mismatching edge gets a Reorder).
    // KEY_LENGTH stays i32 -- reordering it to f32 would numerically convert the index values.
    // sel_indices/sel_count are i32; the state outputs never share memory with their inputs
    // (inPlace == -1): the update is out-of-place.
    const auto data_precision = getOriginalInputPrecisionAtPort(InputIdx::Q_IDX);
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const ov::element::Type precision = i == InputIdx::KEY_LENGTH ? ov::element::i32 : data_precision;
        inPortConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_INDICES), false, -1},
        PortConfigurator{LayoutType::ncsp, ov::element::i32, getOutputShapeAtPort(OutputIdx::SEL_COUNT), false, -1}};
    static constexpr std::pair<size_t, size_t> state_ports[] = {
        {InputIdx::SUMMARY_CACHE, OutputIdx::SUMMARY_CACHE},
        {InputIdx::PAST_INCOMPLETE_RAW_K, OutputIdx::PAST_INCOMPLETE_RAW_K},
        {InputIdx::PAST_INCOMPLETE_ROPE, OutputIdx::PAST_INCOMPLETE_ROPE}};
    for (const auto& [in_port, out_port] : state_ports) {
        outPortConfigs.emplace_back(LayoutType::ncsp, data_precision, getOutputShapeAtPort(out_port), false, -1);
    }
    // The scoring runs on brgemm where the CPU, the precision and the head counts allow it (see createPrimitive()).
    impl_desc_type impl = impl_desc_type::ref;
#ifdef OPENVINO_ARCH_X86_64
    if (brgemm_group_size(*this, data_precision, m_config.h_sel, m_config.indexer_head_dim) != 0) {
        impl = qsa::score_impl_type(data_precision);
    }
#endif
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl);
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto data_precision = getSrcMemoryAtPort(InputIdx::Q_IDX)->getDesc().getPrecision();
    if (data_precision == ov::element::f32) {
        qsa::run_indexer<float>(*this, m_config, context, m_scorer.get());
    } else if (data_precision == ov::element::f16) {
        qsa::run_indexer<ov::float16>(*this, m_config, context, m_scorer.get());
    } else if (data_precision == ov::element::bf16) {
        qsa::run_indexer<ov::bfloat16>(*this, m_config, context, m_scorer.get());
    } else {
        OPENVINO_THROW("QSAIndexer: unsupported precision ", data_precision);
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
        // All floating-point ports share one precision (the fusion pass converts the graph to that).
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
