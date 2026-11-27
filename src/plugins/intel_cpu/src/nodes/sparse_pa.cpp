// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.h"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "nodes/kernels/sparse_attn/paged_page_table_check.hpp"
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"
#include "nodes/paged_attn.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::intel_cpu::node {

namespace {

// The key of the executor in the params cache: everything the executor reads from the configuration of the node. The
// brgemm kernels follow from it (and the CPU), so they are not part of the key. The nodes of a model that share it
// share the executor, and so the buffers of its cache write.
struct ExecutorKey {
    ov::Extensions::Cpu::SparsePAConfig config;

    [[nodiscard]] size_t hash() const {
        using dnnl::impl::hash_combine;
        const auto& c = config;
        size_t seed = 0;
        for (const size_t value : {c.num_heads,
                                   c.num_kv_heads,
                                   c.num_selection_heads,
                                   c.k_head_size,
                                   c.v_head_size,
                                   c.key_block_size,
                                   c.page_size,
                                   c.max_attend_kv_rows}) {
            seed = hash_combine(seed, value);
        }
        seed = hash_combine(seed, c.causal);
        seed = hash_combine(seed, c.data_precision.hash());
        for (const auto* spec : {&c.key_spec, &c.value_spec}) {
            seed = hash_combine(seed, static_cast<size_t>(spec->alg));
            seed = hash_combine(seed, spec->precision.hash());
            seed = hash_combine(seed, spec->group_size);
            seed = hash_combine(seed, spec->by_channel);
        }
        return seed;
    }

    bool operator==(const ExecutorKey& rhs) const {
        const auto& l = config;
        const auto& r = rhs.config;
        return l.num_heads == r.num_heads && l.num_kv_heads == r.num_kv_heads &&
               l.num_selection_heads == r.num_selection_heads && l.k_head_size == r.k_head_size &&
               l.v_head_size == r.v_head_size && l.key_block_size == r.key_block_size && l.page_size == r.page_size &&
               l.max_attend_kv_rows == r.max_attend_kv_rows && l.causal == r.causal &&
               l.data_precision == r.data_precision && l.key_spec == r.key_spec && l.value_spec == r.value_spec;
    }
};

// The executor of a configuration; it builds the AMX kernels of its prefill path (through the params cache) when its
// variant has them.
std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> makeExecutor(const MultiCachePtr& cache,
                                                                    const ov::Extensions::Cpu::SparsePAConfig& config) {
    return ov::Extensions::Cpu::XARCH::make_sparse_pa_executor(config, cache);
}

}  // namespace

SparsePA::SparsePA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto sparse_pa = ov::as_type_ptr<const ov::op::internal::PagedSparseAttention>(op);
    CPU_NODE_ASSERT(sparse_pa, "Expected PagedSparseAttention node");
    if (!deriveConfig(*sparse_pa, m_config, errorMessage)) {
        CPU_NODE_THROW(errorMessage);
    }
}

bool SparsePA::deriveQuant(std::string& reason) {
    const auto key_prec = getOriginalInputPrecisionAtPort(InputIdx::KEY_CACHE);
    const auto value_prec = getOriginalInputPrecisionAtPort(InputIdx::VALUE_CACHE);
    const auto& key_dims = getInputShapeAtPort(InputIdx::KEY_CACHE).getDims();
    const auto& value_dims = getInputShapeAtPort(InputIdx::VALUE_CACHE).getDims();
    m_config.key_spec = {};
    m_config.value_spec = {};
    m_config.page_size = 0;
    if (key_prec != ov::element::u8 && key_prec != ov::element::u4) {
        // Plain caches: the page size is read from the cache shape by the kernel.
        return true;
    }
    for (const auto dim : {key_dims[2], key_dims[3], value_dims[2], value_dims[3]}) {
        if (dim == Shape::UNDEFINED_DIM) {
            reason = "SparsePA CPU implementation requires static page and head dims of quantized caches.";
            return false;
        }
    }
    const auto& cfg = context->getConfig();
    const size_t Dh = m_config.k_head_size;
    const size_t Dv = m_config.v_head_size;
    // Both u8 and u4 store scale and zero point as f32; u4 counts them in half-byte elements.
    const size_t key_params = sizeof(float) * 2 * (key_prec == ov::element::u4 ? 2 : 1);
    const size_t value_params = sizeof(float) * 2 * (value_prec == ov::element::u4 ? 2 : 1);
    auto& key_spec = m_config.key_spec;
    auto& value_spec = m_config.value_spec;
    key_spec.precision = key_prec;
    value_spec.precision = value_prec;
    key_spec.by_channel = PagedAttention::isQuantByChannel(cfg.keyCacheQuantMode, key_prec, true);
    // A group size of 0 means one group per row; PagedAttention resolves it the same way before quantizing.
    key_spec.group_size = cfg.keyCacheGroupSize ? cfg.keyCacheGroupSize : Dh;
    value_spec.group_size = cfg.valueCacheGroupSize ? cfg.valueCacheGroupSize : Dv;
    if (PagedAttention::isQuantByChannel(cfg.valueCacheQuantMode, value_prec, false)) {
        reason = "SparsePA CPU implementation does not support by-channel quantized value caches.";
        return false;
    }
    if (value_prec != ov::element::u8 && value_prec != ov::element::u4) {
        reason = "SparsePA CPU implementation requires the key and value caches to be both quantized or both plain.";
        return false;
    }

    // The page size is key_cache dim 2 (for by-channel keys dim 2 also holds the parameter rows, which are subtracted).
    if (key_spec.by_channel) {
        if (key_dims[3] != Dh || key_dims[2] <= key_params || (key_prec == ov::element::u4 && Dh % 2 != 0)) {
            reason = "SparsePA: by-channel key cache shape does not match the head size and quantization parameters.";
            return false;
        }
        m_config.page_size = key_dims[2] - key_params;
    } else {
        const size_t gs = key_spec.group_size;
        if (Dh % gs != 0 || key_dims[3] != Dh + key_params * (Dh / gs)) {
            reason = "SparsePA: by-token key cache shape does not match the head size and group size.";
            return false;
        }
        m_config.page_size = key_dims[2];
    }
    const size_t vgs = value_spec.group_size;
    if (Dv % vgs != 0 || value_dims[3] != Dv + value_params * (Dv / vgs) || value_dims[2] != m_config.page_size) {
        reason = "SparsePA: value cache shape does not match the head size, group size and key page size.";
        return false;
    }
    if (m_config.page_size % m_config.key_block_size != 0) {
        reason = "SparsePA: the cache page size must be divisible by key_block_size.";
        return false;
    }
    return true;
}

bool SparsePA::deriveConfig(const ov::op::internal::PagedSparseAttention& op, Config& config, std::string& reason) {
    const auto& query_ps = op.get_input_partial_shape(InputIdx::QUERY);
    const auto& key_ps = op.get_input_partial_shape(InputIdx::KEY);
    const auto& value_ps = op.get_input_partial_shape(InputIdx::VALUE);
    const auto& key_cache_ps = op.get_input_partial_shape(InputIdx::KEY_CACHE);
    const auto& value_cache_ps = op.get_input_partial_shape(InputIdx::VALUE_CACHE);
    const auto& selected_ps = op.get_input_partial_shape(InputIdx::SELECTED_BLOCK_INDICES);
    const auto rank_is = [](const ov::PartialShape& ps, int64_t rank) {
        return ps.rank().is_static() && ps.rank().get_length() == rank;
    };
    if (!rank_is(query_ps, 2) || !rank_is(key_ps, 2) || !rank_is(value_ps, 2) || !rank_is(key_cache_ps, 4) ||
        !rank_is(value_cache_ps, 4) || !rank_is(selected_ps, 3) || !query_ps[1].is_static() || !key_ps[1].is_static() ||
        !value_ps[1].is_static() || !selected_ps[1].is_static()) {
        reason = "PagedSparseAttention CPU implementation requires static head dimensions.";
        return false;
    }
    const auto query_width = static_cast<size_t>(query_ps[1].get_length());
    const auto key_width = static_cast<size_t>(key_ps[1].get_length());
    const auto value_width = static_cast<size_t>(value_ps[1].get_length());
    config.num_selection_heads = static_cast<size_t>(selected_ps[1].get_length());
    // The head counts and sizes are fixed by the model. The transformation that creates the op records them in the
    // runtime info (like PagedAttention), because the cache shapes are padded when the caches are quantized.
    const auto& rt = op.get_rt_info();
    for (const char* name : {"num_k_heads", "k_head_size", "num_v_heads", "v_head_size"}) {
        if (rt.count(name) == 0UL) {
            reason = std::string("PagedSparseAttention CPU implementation requires the runtime info ") + name;
            return false;
        }
    }
    config.num_kv_heads = rt.at("num_k_heads").as<size_t>();
    config.k_head_size = rt.at("k_head_size").as<size_t>();
    config.v_head_size = rt.at("v_head_size").as<size_t>();
    if (config.num_kv_heads == 0 || rt.at("num_v_heads").as<size_t>() != config.num_kv_heads ||
        key_width != config.num_kv_heads * config.k_head_size ||
        value_width != config.num_kv_heads * config.v_head_size) {
        reason =
            "PagedSparseAttention: the key and value widths must match the head counts and sizes of the runtime info.";
        return false;
    }
    if (config.k_head_size == 0 || config.v_head_size == 0 || query_width % config.k_head_size != 0) {
        reason = "PagedSparseAttention: the query width must be a multiple of the key head size.";
        return false;
    }
    config.num_heads = query_width / config.k_head_size;
    if (config.num_heads == 0 || config.num_heads % config.num_kv_heads != 0) {
        reason = "PagedSparseAttention: the number of query heads must be a multiple of the number of KV heads.";
        return false;
    }
    config.key_block_size = static_cast<size_t>(op.get_key_block_size());
    config.causal = true;
    if (config.num_selection_heads != 1 && config.num_selection_heads != config.num_kv_heads &&
        config.num_selection_heads != config.num_heads) {
        reason = "PagedSparseAttention CPU implementation requires 1, num_kv_heads, or num_heads selection heads.";
        return false;
    }
    return true;
}

void SparsePA::createPrimitive() {
    [[maybe_unused]] const auto Dh = m_config.k_head_size;
    // The page size is KEY_CACHE's dim 2; if it is dynamic the page / key_block_size fit can't be decided.
    // By-channel quantized key pages carry parameter rows in dim 2, so quantized caches use the derived page size.
    const auto page_size = isQuantized() ? m_config.page_size : getInputShapeAtPort(InputIdx::KEY_CACHE).getDims()[2];
    CPU_NODE_ASSERT(page_size == Shape::UNDEFINED_DIM || page_size % m_config.key_block_size == 0,
                    "the cache page size ",
                    page_size,
                    " must be divisible by key_block_size ",
                    m_config.key_block_size);
    // The kernel attends adjacent selected blocks of one page together, up to a page of rows; with a dynamic page size
    // every block is attended on its own.
    m_config.max_attend_kv_rows = page_size != Shape::UNDEFINED_DIM ? page_size : m_config.key_block_size;
    // The kernel reads QUERY, KEY, VALUE and writes the output as one data type.
    m_config.data_precision = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
    CPU_NODE_ASSERT(getOriginalInputPrecisionAtPort(InputIdx::KEY) == m_config.data_precision &&
                        getOriginalInputPrecisionAtPort(InputIdx::VALUE) == m_config.data_precision &&
                        getOriginalOutputPrecisionAtPort(0) == m_config.data_precision,
                    "query, key, value and output must have one precision, but the query is ",
                    m_config.data_precision);
    // The executor is shared by the nodes of the same configuration, see ExecutorKey.
    const auto cache = context->getParamsCache();
    [[maybe_unused]] const auto [sparse_pa_executor, lookup_status] =
        cache->getOrCreate(ExecutorKey{m_config}, [&cache](const ExecutorKey& key) {
            return makeExecutor(cache, key.config);
        });
    CPU_NODE_ASSERT(sparse_pa_executor, "executor creation fails");
    m_executor = sparse_pa_executor;
    getSelectedPrimitiveDescriptor()->setImplementationType(m_executor->impl_type());
}

void SparsePA::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // The cache layout depends on the cache precisions, which are final only now (after EnforceInferencePrecision).
    std::string errorMessage;
    if (!deriveQuant(errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    // Float ports keep the precision settled upstream: the caches grow every step, so a Reorder would convert the
    // whole history each call. The page table and selection inputs are i32; SCALE keeps its own precision and is
    // converted to f32 when read. Plain caches: the query type, or f16 next to f32 data like PagedAttention.
    if (!isQuantized()) {
        const auto query_prec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        const auto cache_prec = getOriginalInputPrecisionAtPort(InputIdx::KEY_CACHE);
        CPU_NODE_ASSERT(cache_prec == query_prec || (query_prec == ov::element::f32 && cache_prec == ov::element::f16),
                        "plain caches must have the query precision, or f16 with an f32 query, but got ",
                        cache_prec,
                        " caches with a ",
                        query_prec,
                        " query");
    }
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_i32_port = i == InputIdx::BLOCK_INDICES || i == InputIdx::BLOCK_INDICES_BEGINS ||
                                 i == InputIdx::SUBSEQUENCE_BEGINS || i == InputIdx::PAST_LENS ||
                                 i == InputIdx::SELECTED_BLOCK_INDICES || i == InputIdx::SELECTED_BLOCK_COUNT;
        const ov::element::Type precision = is_i32_port ? ov::element::i32 : getOriginalInputPrecisionAtPort(i);
        inPortConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void SparsePA::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto& c = m_config;

    PlainTensor query(getSrcMemoryAtPort(InputIdx::QUERY));
    PlainTensor key(getSrcMemoryAtPort(InputIdx::KEY));
    PlainTensor value(getSrcMemoryAtPort(InputIdx::VALUE));
    PlainTensor key_cache(getSrcMemoryAtPort(InputIdx::KEY_CACHE));
    PlainTensor value_cache(getSrcMemoryAtPort(InputIdx::VALUE_CACHE));
    PlainTensor block_indices(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES));
    PlainTensor block_indices_begins(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES_BEGINS));
    PlainTensor subsequence_begins(getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    PlainTensor past_lens(getSrcMemoryAtPort(InputIdx::PAST_LENS));
    PlainTensor sel_indices(getSrcMemoryAtPort(InputIdx::SELECTED_BLOCK_INDICES));
    PlainTensor sel_count(getSrcMemoryAtPort(InputIdx::SELECTED_BLOCK_COUNT));
    float scale = 0.0F;
    switch (getParentEdgeAt(InputIdx::SCALE)->getMemory().getPrecision()) {
    case ov::element::bf16:
        scale = static_cast<float>(*getSrcDataAtPortAs<const ov::bfloat16>(InputIdx::SCALE));
        break;
    case ov::element::f16:
        scale = static_cast<float>(*getSrcDataAtPortAs<const ov::float16>(InputIdx::SCALE));
        break;
    default:
        scale = *getSrcDataAtPortAs<const float>(InputIdx::SCALE);
        break;
    }
    PlainTensor out(getDstMemoryAtPort(0));
    const auto& cpu_parallel = context->getCpuParallel();

    const size_t block_size = m_config.page_size ? m_config.page_size : key_cache.size(2);
    CPU_NODE_ASSERT(block_size % c.key_block_size == 0,
                    "the cache page size ",
                    block_size,
                    " must be divisible by key_block_size ",
                    c.key_block_size);
    // Core shape inference leaves the cache layout to the backend: the caches must match the config, or the K/V write
    // and the kernel would write / read out of bounds. A quantized cache layout (dims 2 and 3) is checked in
    // deriveQuant.
    if (isQuantized()) {
        key_cache.assert_dims({0, c.num_kv_heads, 0, 0}, true);
        value_cache.assert_dims({key_cache.size(0), c.num_kv_heads, 0, 0}, true);
    } else {
        key_cache.assert_dims({0, c.num_kv_heads, block_size, c.k_head_size}, true);
        value_cache.assert_dims({key_cache.size(0), c.num_kv_heads, block_size, c.v_head_size}, true);
    }
    check_paged_page_table(past_lens,
                           subsequence_begins,
                           block_indices,
                           block_indices_begins,
                           query.size(0),
                           block_size,
                           std::min(key_cache.size(0), value_cache.size(0)));

    const ov::Extensions::Cpu::SparsePATensors tensors{query,
                                                       key,
                                                       value,
                                                       key_cache,
                                                       value_cache,
                                                       block_indices,
                                                       block_indices_begins,
                                                       subsequence_begins,
                                                       past_lens,
                                                       sel_indices,
                                                       sel_count,
                                                       out,
                                                       scale};
    // The executor writes the current K/V into the caches (parallel over token x kv head), then runs the sparse
    // attention (parallel over query token x query head pack) with per-thread scratch.
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8,
                                               ov::intel_cpu::Shape{m_executor->scratch_bytes(tensors, cpu_parallel)}));
    m_executor->execute(tensors, scratch, cpu_parallel);
}

bool SparsePA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto sparse_pa = ov::as_type_ptr<const ov::op::internal::PagedSparseAttention>(op);
        if (!sparse_pa) {
            errorMessage = "Only PagedSparseAttention operation is supported";
            return false;
        }
        for (const auto port : {InputIdx::QUERY, InputIdx::KEY, InputIdx::VALUE}) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "SparsePA CPU implementation only supports f32, f16, or bf16 query, key and value.";
                return false;
            }
        }
        // The caches are either both plain f32/f16/bf16 or both quantized to u8/u4 (see CacheSpec).
        const auto is_plain = [](const ov::element::Type& prec) {
            return prec == ov::element::f32 || prec == ov::element::f16 || prec == ov::element::bf16;
        };
        const auto is_quantized = [](const ov::element::Type& prec) {
            return prec == ov::element::u8 || prec == ov::element::u4;
        };
        const auto key_cache_prec = op->get_input_element_type(InputIdx::KEY_CACHE);
        const auto value_cache_prec = op->get_input_element_type(InputIdx::VALUE_CACHE);
        // Whether the plain cache type fits the query type is checked on the precisions the graph settles on (see
        // initSupportedPrimitiveDescriptors): the inference precision may still change the query type.
        if (is_plain(key_cache_prec) && key_cache_prec != value_cache_prec) {
            errorMessage = "SparsePA CPU implementation requires the plain key and value caches to have one type.";
            return false;
        }
        if ((!is_plain(key_cache_prec) || !is_plain(value_cache_prec)) &&
            (!is_quantized(key_cache_prec) || !is_quantized(value_cache_prec))) {
            errorMessage =
                "SparsePA CPU implementation requires both caches to be plain f32/f16/bf16 or both quantized to u8/u4.";
            return false;
        }
        // The op already requires the page table and selection plan to be i32 (or dynamic).
        static constexpr size_t i32_ports[] = {InputIdx::BLOCK_INDICES,
                                               InputIdx::BLOCK_INDICES_BEGINS,
                                               InputIdx::SUBSEQUENCE_BEGINS,
                                               InputIdx::PAST_LENS,
                                               InputIdx::SELECTED_BLOCK_INDICES,
                                               InputIdx::SELECTED_BLOCK_COUNT};
        for (auto port : i32_ports) {
            if (op->get_input_element_type(port) != ov::element::i32) {
                errorMessage = "SparsePA CPU implementation requires i32 index inputs.";
                return false;
            }
        }
        // Sinks are not supported yet: SINKS must be an empty tensor.
        const auto& sinks_ps = op->get_input_partial_shape(InputIdx::SINKS);
        if (!sinks_ps.is_static() || ov::shape_size(sinks_ps.to_shape()) != 0) {
            errorMessage = "SparsePA CPU implementation does not support sinks.";
            return false;
        }
        Config config;
        if (!deriveConfig(*sparse_pa, config, errorMessage)) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
