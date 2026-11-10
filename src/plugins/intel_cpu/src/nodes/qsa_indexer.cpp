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
    m_impl = qsa::kernel_impl_type();
#ifdef OPENVINO_ARCH_X86_64
    // The scoring on brgemm where the CPU, the precision and the head counts allow it; the head counts are static, the
    // batch and the window are not. The scorer is shared by the layers of the same configuration.
    const auto& c = m_config;
    const size_t q_heads = getInputShapeAtPort(InputIdx::Q_IDX).getDims()[2];
    const size_t k_heads = getInputShapeAtPort(InputIdx::K_IDX).getDims()[2];
    if (q_heads != Shape::UNDEFINED_DIM && k_heads != Shape::UNDEFINED_DIM) {
        m_brgemm = qsa::QsaBrgemm::get(
            context->getParamsCache(),
            qsa::QsaBrgemm::Config{q_heads,
                                   k_heads,
                                   c.h_sel,
                                   c.indexer_head_dim,
                                   c.indexer_rotary_dim,
                                   c.block_topk,
                                   static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()),
                                   c.eps,
                                   QSAIndexerNode::effective_scale(c),
                                   precision,
                                   c.order_by_score});
    }
    if (m_brgemm) {
        m_impl = qsa::score_impl_type(precision);
    }
#endif
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_impl);
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = getSrcMemoryAtPort(i);
    }
    std::vector<MemoryPtr> outputs(OutputIdx::COUNT);
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = getDstMemoryAtPort(i);
    }
    const qsa::DensePorts ports(inputs, outputs);
    const auto cpu_parallel = context->getCpuParallel();

    // The calls that have complete blocks to score go to the brgemm scorer, which plans its scratch from the sizes of
    // the call; the other calls are scored by the kernel itself.
    const qsa::QsaBrgemm* brgemm = nullptr;
    size_t scorer_shared_bytes = 0;
    size_t scorer_thread_bytes = 0;
#ifdef OPENVINO_ARCH_X86_64
    if (m_brgemm && ports.summary_cache.size(2) > 0) {
        const auto stats = qsa::row_stats(ports, m_config);
        const auto plan = m_brgemm->plan(ports.q_idx.size(0), stats.max_current, stats.max_row_blocks);
        brgemm = m_brgemm.get();
        scorer_shared_bytes = plan.shared_bytes;
        scorer_thread_bytes = plan.thread_bytes;
    }
#endif
    const qsa::DenseScratchLayout layout(ports,
                                         m_config,
                                         scorer_shared_bytes,
                                         scorer_thread_bytes,
                                         static_cast<size_t>(cpu_parallel->get_num_worker_threads()));
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{layout.total_bytes}));

    ov::Extensions::Cpu::XARCH::qsa_indexer_compute(ports,
                                                    m_config,
                                                    layout,
                                                    scratch->getDataAs<uint8_t>(),
                                                    brgemm,
                                                    cpu_parallel);
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
