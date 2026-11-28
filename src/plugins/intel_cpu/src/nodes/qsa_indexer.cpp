// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
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
#include "nodes/kernels/sparse_attn/qsa_executor_cache.hpp"
#include "nodes/kernels/sparse_attn/qsa_indexer_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::QSAIndexer::InputIdx;
using OutputIdx = ov::op::internal::QSAIndexer::OutputIdx;

// Output shapes of QSAIndexer: everything follows the input shapes, but the block_index_keys_cache block count of the
// longest row, which is read from the key_length values (the op reports the input shape: FuseQSAIndexer without padding
// leaves the cache growth to the implementation, see its documentation).
class QSAIndexerShapeInfer : public ShapeInferEmptyPads {
public:
    explicit QSAIndexerShapeInfer(const ov::op::internal::QSAIndexer::Config& config) : m_config(config) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& q_dims = input_shapes[InputIdx::INDEX_QUERY].get();
        const auto& k_dims = input_shapes[InputIdx::RAW_INDEX_KEYS].get();
        const auto& cache_dims = input_shapes[InputIdx::BLOCK_INDEX_KEYS_CACHE].get();
        const size_t batch = q_dims[0];
        const size_t window = q_dims[1];
        const size_t head_dim = m_config.indexer_head_dim;
        const size_t rotary_dim = m_config.indexer_rotary_dim;
        const size_t key_heads = k_dims[2];
        // The kernels index every tensor with the batch of the query and the sizes of the config, without bounds
        // checks, and the core op checks the shapes only at compile time, when the batch is dynamic.
        const auto same_dims = [&](size_t idx, std::initializer_list<size_t> expected) {
            return std::equal(input_shapes[idx].get().begin(),
                              input_shapes[idx].get().end(),
                              expected.begin(),
                              expected.end());
        };
        const auto rope_ok = [&](size_t idx) {
            const auto& dims = input_shapes[idx].get();
            return dims.size() == 3 && (dims[0] == batch || dims[0] == 1) && dims[1] == window && dims[2] == rotary_dim;
        };
        const size_t key_length_count = input_shapes[InputIdx::KEY_LENGTH].get()[0];
        OPENVINO_ASSERT(q_dims[3] == head_dim && k_dims.size() == 4 && k_dims[0] == batch && k_dims[1] == window &&
                            k_dims[3] == head_dim,
                        "QSAIndexer: index_query and raw_index_keys must be [B, L, H, ",
                        head_dim,
                        "] with the same batch and window");
        OPENVINO_ASSERT(rope_ok(InputIdx::ROPE_COS) && rope_ok(InputIdx::ROPE_SIN),
                        "QSAIndexer: rope_cos and rope_sin must be [B or 1, L, ",
                        rotary_dim,
                        "]");
        OPENVINO_ASSERT(input_shapes[InputIdx::KEY_LENGTH].get().size() == 1 &&
                            (key_length_count == 1 || key_length_count == batch),
                        "QSAIndexer: key_length must have 1 or B entries");
        OPENVINO_ASSERT(
            same_dims(InputIdx::Q_NORM_WEIGHT, {head_dim}) && same_dims(InputIdx::K_NORM_WEIGHT, {head_dim}),
            "QSAIndexer: the norm weights must be [",
            head_dim,
            "]");
        OPENVINO_ASSERT(
            same_dims(InputIdx::PAST_INCOMPLETE_RAW_K, {batch, key_heads, m_config.key_block_size - 1, head_dim}),
            "QSAIndexer: past_incomplete_raw_k must be [B, H_ik, key_block_size - 1, ",
            head_dim,
            "]");
        OPENVINO_ASSERT(same_dims(InputIdx::PAST_INCOMPLETE_ROPE, {batch, 2 * rotary_dim}),
                        "QSAIndexer: past_incomplete_rope must be [B, 2 * ",
                        rotary_dim,
                        "]");
        OPENVINO_ASSERT(
            cache_dims.size() == 4 && cache_dims[0] == batch && cache_dims[1] == key_heads && cache_dims[3] == head_dim,
            "QSAIndexer: block_index_keys_cache must be [B, H_ik, blocks, ",
            head_dim,
            "]");
        const auto& key_length = data_dependency.at(InputIdx::KEY_LENGTH);
        const auto* key_length_p = key_length->getDataAs<const int32_t>();
        // key_length is [batch] or [1] (the same length for all the rows)
        int64_t longest = 0;
        for (size_t b = 0; b < key_length_count; b++) {
            longest = std::max<int64_t>(longest, key_length_p[b]);
        }
        const auto num_blocks = static_cast<size_t>(longest / static_cast<int64_t>(m_config.key_block_size));
        return {{{batch, m_config.num_selection_heads, q_dims[1], m_config.block_topk},
                 {batch, m_config.num_selection_heads, q_dims[1]},
                 {batch, cache_dims[1], num_blocks, cache_dims[3]},
                 input_shapes[InputIdx::PAST_INCOMPLETE_RAW_K].get(),
                 input_shapes[InputIdx::PAST_INCOMPLETE_ROPE].get()},
                ShapeInferStatus::success};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return 1U << InputIdx::KEY_LENGTH;
    }

private:
    ov::op::internal::QSAIndexer::Config m_config;
};

class QSAIndexerShapeInferFactory : public ShapeInferFactory {
public:
    explicit QSAIndexerShapeInferFactory(const std::shared_ptr<ov::Node>& op) {
        if (const auto qsa_indexer = ov::as_type_ptr<const ov::op::internal::QSAIndexer>(op)) {
            m_config = qsa_indexer->get_config();
        }
    }

    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<QSAIndexerShapeInfer>(m_config);
    }

private:
    ov::op::internal::QSAIndexer::Config m_config;
};

}  // namespace

QSAIndexer::QSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, QSAIndexerShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto qsa_indexer = ov::as_type_ptr<const ov::op::internal::QSAIndexer>(op);
    CPU_NODE_ASSERT(qsa_indexer, "Expected QSAIndexer node");
    m_config = qsa_indexer->get_config();
    // Fixed by the model (isSupportedOperation checked they are static).
    m_index_query_heads = qsa_indexer->get_index_query_heads();
    m_index_key_heads = qsa_indexer->get_index_key_heads();
}

void QSAIndexer::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    const auto data_precision = getOriginalInputPrecisionAtPort(InputIdx::INDEX_QUERY);

    std::vector<PortConfigurator> in_port_configs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const ov::element::Type precision = i == InputIdx::KEY_LENGTH ? ov::element::i32 : data_precision;
        in_port_configs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> out_port_configs;
    for (size_t i = 0; i < OutputIdx::COUNT; ++i) {
        const bool is_index = i == OutputIdx::SELECTED_BLOCK_INDICES || i == OutputIdx::SELECTED_BLOCK_COUNT;
        out_port_configs.emplace_back(LayoutType::ncsp,
                                      is_index ? ov::element::i32 : data_precision,
                                      getOutputShapeAtPort(i),
                                      false,
                                      -1);
    }
    addSupportedPrimDesc(in_port_configs, out_port_configs, impl_desc_type::ref_any);
}

void QSAIndexer::createPrimitive() {
    const auto precision = getSrcMemoryAtPort(InputIdx::INDEX_QUERY)->getDesc().getPrecision();
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
    config.scale = ov::op::internal::QSAIndexer::effective_scale(c);
    config.precision = precision;
    config.query_norm = c.query_norm;
    config.query_rope = c.query_rope;
    // The executor is shared by the layers of the same configuration.
    const auto cache = context->getParamsCache();
    [[maybe_unused]] const auto [qsa_indexer_executor, lookup_status] =
        cache->getOrCreate(ov::Extensions::Cpu::qsa::QsaExecutorKey{config},
                           [&cache](const ov::Extensions::Cpu::qsa::QsaExecutorKey& key) {
                               return ov::Extensions::Cpu::XARCH::make_indexer_executor(key.config, cache);
                           });
    CPU_NODE_ASSERT(qsa_indexer_executor, "executor creation fails");
    m_executor = qsa_indexer_executor;
    Node::createPrimitive();
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->impl_type());
}

void QSAIndexer::execute([[maybe_unused]] const dnnl::stream& strm) {
    ov::Extensions::Cpu::qsa::IndexerArgs call;
    call.q_idx.reset(getSrcMemoryAtPort(InputIdx::INDEX_QUERY));
    call.k_idx.reset(getSrcMemoryAtPort(InputIdx::RAW_INDEX_KEYS));
    call.rope_cos.reset(getSrcMemoryAtPort(InputIdx::ROPE_COS));
    call.rope_sin.reset(getSrcMemoryAtPort(InputIdx::ROPE_SIN));
    call.key_length.reset(getSrcMemoryAtPort(InputIdx::KEY_LENGTH));
    call.q_norm_weight.reset(getSrcMemoryAtPort(InputIdx::Q_NORM_WEIGHT));
    call.k_norm_weight.reset(getSrcMemoryAtPort(InputIdx::K_NORM_WEIGHT));
    // The state inputs are read-only; the updated state lives in the matching outputs.
    call.past_raw_k_in.reset(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_RAW_K));
    call.past_rope_in.reset(getSrcMemoryAtPort(InputIdx::PAST_INCOMPLETE_ROPE));
    call.summary_cache_in.reset(getSrcMemoryAtPort(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    call.summary_cache.reset(getDstMemoryAtPort(OutputIdx::BLOCK_INDEX_KEYS_CACHE));
    call.past_raw_k.reset(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_RAW_K));
    call.past_rope.reset(getDstMemoryAtPort(OutputIdx::PAST_INCOMPLETE_ROPE));
    call.sel_indices.reset(getDstMemoryAtPort(OutputIdx::SELECTED_BLOCK_INDICES));
    call.sel_count.reset(getDstMemoryAtPort(OutputIdx::SELECTED_BLOCK_COUNT));

    m_executor->find_batch_max_params(call);
    // One scratch for all the stages: a second createScratchPadMem() could free the memory of the first.
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{m_executor->scratch_bytes(call)}));
    m_executor->execute(call, scratch->getDataAs<uint8_t>(), context->getCpuParallel());
}

bool QSAIndexer::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto qsa_indexer = ov::as_type_ptr<const ov::op::internal::QSAIndexer>(op);
        if (!qsa_indexer) {
            errorMessage = "Only QSAIndexer operation is supported";
            return false;
        }
        // The head counts are fixed once, when the executor is built; the batch and the window may change.
        const auto& query_ps = op->get_input_partial_shape(InputIdx::INDEX_QUERY);
        const auto& key_ps = op->get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
        if (query_ps.rank().is_dynamic() || query_ps.size() != 4 || query_ps[2].is_dynamic() ||
            key_ps.rank().is_dynamic() || key_ps.size() != 4 || key_ps[2].is_dynamic()) {
            errorMessage = "QSAIndexer CPU implementation requires static head counts.";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::INDEX_QUERY,
                                                 InputIdx::RAW_INDEX_KEYS,
                                                 InputIdx::ROPE_COS,
                                                 InputIdx::ROPE_SIN,
                                                 InputIdx::Q_NORM_WEIGHT,
                                                 InputIdx::K_NORM_WEIGHT,
                                                 InputIdx::BLOCK_INDEX_KEYS_CACHE,
                                                 InputIdx::PAST_INCOMPLETE_RAW_K,
                                                 InputIdx::PAST_INCOMPLETE_ROPE};
        // All floating-point ports share one precision (the op producer sets it).
        const auto data_precision = op->get_input_element_type(InputIdx::INDEX_QUERY);
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
        if (!ov::op::internal::QSAIndexer::validate_config(qsa_indexer->get_config(), reason)) {
            errorMessage = "QSAIndexer configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
