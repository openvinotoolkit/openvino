// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/attn_memcpy.hpp"
#include "nodes/kernels/scaled_attn/attn_quant.hpp"
#include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#include "nodes/kernels/sparse_attn/sparse_pa_kernel.hpp"
#include "nodes/paged_attn.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include <cpu/x64/cpu_isa_traits.hpp>

#    include "nodes/kernels/sparse_attn/sparse_kernel_cache.hpp"
#endif

namespace ov::intel_cpu::node {

SparsePA::SparsePA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto sparse_pa = ov::as_type_ptr<const ov::op::internal::PagedSparseAttention>(op);
    OPENVINO_ASSERT(sparse_pa, "Expected PagedSparseAttention node");
    if (!deriveConfig(*sparse_pa, m_config, errorMessage)) {
        OPENVINO_THROW(errorMessage);
    }
}

bool SparsePA::deriveQuant(std::string& reason) {
    const auto key_prec = getOriginalInputPrecisionAtPort(InputIdx::KEY_CACHE);
    const auto value_prec = getOriginalInputPrecisionAtPort(InputIdx::VALUE_CACHE);
    const auto& key_dims = getInputShapeAtPort(InputIdx::KEY_CACHE).getDims();
    const auto& value_dims = getInputShapeAtPort(InputIdx::VALUE_CACHE).getDims();
    m_quant = {};
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
    m_quant.key_prec = key_prec;
    m_quant.value_prec = value_prec;
    m_quant.key_bychannel = PagedAttention::isQuantByChannel(cfg.keyCacheQuantMode, key_prec, true);
    // A group size of 0 means one group per row; PagedAttention resolves it the same way before quantizing.
    m_quant.key_group_size = cfg.keyCacheGroupSize ? cfg.keyCacheGroupSize : Dh;
    m_quant.value_group_size = cfg.valueCacheGroupSize ? cfg.valueCacheGroupSize : Dv;
    if (PagedAttention::isQuantByChannel(cfg.valueCacheQuantMode, value_prec, false)) {
        reason = "SparsePA CPU implementation does not support by-channel quantized value caches.";
        return false;
    }
    if (value_prec != ov::element::u8 && value_prec != ov::element::u4) {
        reason = "SparsePA CPU implementation requires the key and value caches to be both quantized or both plain.";
        return false;
    }

    // The page size is key_cache dim 2, plus parameter rows for by-channel keys.
    if (m_quant.key_bychannel) {
        if (key_dims[3] != Dh || key_dims[2] <= key_params || (key_prec == ov::element::u4 && Dh % 2 != 0)) {
            reason = "SparsePA: by-channel key cache shape does not match the head size and quantization parameters.";
            return false;
        }
        m_quant.block_size = key_dims[2] - key_params;
    } else {
        const size_t gs = m_quant.key_group_size;
        if (Dh % gs != 0 || key_dims[3] != Dh + key_params * (Dh / gs)) {
            reason = "SparsePA: by-token key cache shape does not match the head size and group size.";
            return false;
        }
        m_quant.block_size = key_dims[2];
    }
    const size_t vgs = m_quant.value_group_size;
    if (Dv % vgs != 0 || value_dims[3] != Dv + value_params * (Dv / vgs) || value_dims[2] != m_quant.block_size) {
        reason = "SparsePA: value cache shape does not match the head size, group size and key page size.";
        return false;
    }
    if (m_quant.block_size % m_config.key_block_size != 0) {
        reason = "SparsePA: the cache page size must be divisible by key_block_size.";
        return false;
    }
    return true;
}

size_t SparsePA::writeKVSlotMappingBytes(size_t tokens) {
    return ov::Extensions::Cpu::aligned(tokens * sizeof(int32_t));
}

size_t SparsePA::writeKVScratchBytes(size_t tokens, size_t block_size) const {
    const auto& c = m_config;
    size_t bytes = writeKVSlotMappingBytes(tokens);
    if (m_quant.enabled() && m_quant.key_bychannel) {
        bytes += static_cast<size_t>(parallel_get_max_threads()) * block_size * c.num_heads *
                 std::max(c.k_head_size, c.v_head_size) * sizeof(float);
    }
    return bytes;
}

void SparsePA::writeKV(uint8_t* scratch) {
    PlainTensor key(getSrcMemoryAtPort(InputIdx::KEY));
    PlainTensor value(getSrcMemoryAtPort(InputIdx::VALUE));
    PlainTensor key_cache(getSrcMemoryAtPort(InputIdx::KEY_CACHE));
    PlainTensor value_cache(getSrcMemoryAtPort(InputIdx::VALUE_CACHE));
    PlainTensor past_lens(getSrcMemoryAtPort(InputIdx::PAST_LENS));
    PlainTensor subsequence_begins(getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    PlainTensor block_indices(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES));
    PlainTensor block_indices_begins(getSrcMemoryAtPort(InputIdx::BLOCK_INDICES_BEGINS));
    const auto& c = m_config;
    const size_t block_size = m_quant.enabled() ? m_quant.block_size : key_cache.size(2);

    // Physical slot (block * block_size + offset) of every new token, as PagedAttention builds it.
    const size_t tokens = key.size(0);
    PlainTensor slot_mapping;
    slot_mapping.resize<int32_t>({tokens}, reinterpret_cast<int32_t*>(scratch));
    size_t idx = 0;
    for (size_t seq = 0; seq < past_lens.size(0); seq++) {
        const int32_t q_len = subsequence_begins.ptr<int32_t>()[seq + 1] - subsequence_begins.ptr<int32_t>()[seq];
        const int32_t past = past_lens.ptr<int32_t>()[seq];
        const int32_t first_block = block_indices_begins.ptr<int32_t>()[seq];
        for (int32_t j = 0; j < q_len; j++) {
            const auto pos = static_cast<size_t>(past) + static_cast<size_t>(j);
            const auto block = static_cast<size_t>(block_indices.ptr<int32_t>()[first_block + pos / block_size]);
            slot_mapping.ptr<int32_t>()[idx++] = static_cast<int32_t>(block * block_size + pos % block_size);
        }
    }

    const auto key_4d = key.reshape({tokens, c.num_kv_heads, 1, c.k_head_size});
    const auto value_4d = value.reshape({tokens, c.num_kv_heads, 1, c.v_head_size});
    if (!m_quant.enabled()) {
        ov::Extensions::Cpu::XARCH::paged_attn_memcpy(key_4d,
                                                      value_4d,
                                                      key_cache,
                                                      value_cache,
                                                      slot_mapping,
                                                      context->getCpuParallel());
        return;
    }
    ov::Extensions::Cpu::QuantizeParam quant_param{m_quant.key_bychannel,
                                                   false,
                                                   false,
                                                   m_quant.key_group_size,
                                                   m_quant.value_group_size};
    PlainTensor temp_buffer;
    if (m_quant.key_bychannel) {
        temp_buffer.resize<float>({static_cast<size_t>(parallel_get_max_threads()),
                                   block_size,
                                   c.num_heads,
                                   std::max(c.k_head_size, c.v_head_size)},
                                  reinterpret_cast<float*>(scratch + writeKVSlotMappingBytes(tokens)));
    }
    ov::Extensions::Cpu::XARCH::paged_attn_quantkv(key_4d,
                                                   value_4d,
                                                   key_cache,
                                                   value_cache,
                                                   past_lens,
                                                   subsequence_begins,
                                                   block_indices,
                                                   block_indices_begins,
                                                   slot_mapping,
                                                   temp_buffer,
                                                   quant_param,
                                                   context->getCpuParallel());
}

bool SparsePA::deriveConfig(const ov::op::internal::PagedSparseAttention& op, Config& config, std::string& reason) {
    const auto& query_ps = op.get_input_partial_shape(InputIdx::QUERY);
    const auto& key_ps = op.get_input_partial_shape(InputIdx::KEY);
    const auto& value_ps = op.get_input_partial_shape(InputIdx::VALUE);
    const auto& key_cache_ps = op.get_input_partial_shape(InputIdx::KEY_CACHE);
    const auto& selected_ps = op.get_input_partial_shape(InputIdx::SELECTED_BLOCK_INDICES);
    const auto rank_is = [](const ov::PartialShape& ps, int64_t rank) {
        return ps.rank().is_static() && ps.rank().get_length() == rank;
    };
    if (!rank_is(query_ps, 2) || !rank_is(key_ps, 2) || !rank_is(value_ps, 2) || !rank_is(key_cache_ps, 4) ||
        !rank_is(selected_ps, 3) || !query_ps[1].is_static() || !key_ps[1].is_static() || !value_ps[1].is_static() ||
        !key_cache_ps[1].is_static() || !selected_ps[1].is_static()) {
        reason = "PagedSparseAttention CPU implementation requires static head dimensions.";
        return false;
    }
    const auto query_width = static_cast<size_t>(query_ps[1].get_length());
    const auto key_width = static_cast<size_t>(key_ps[1].get_length());
    const auto value_width = static_cast<size_t>(value_ps[1].get_length());
    config.num_kv_heads = static_cast<size_t>(key_cache_ps[1].get_length());
    config.num_selection_heads = static_cast<size_t>(selected_ps[1].get_length());
    // The head sizes are derived from the widths of the flattened inputs: they must split evenly.
    if (config.num_kv_heads == 0 || key_width % config.num_kv_heads != 0 || value_width % config.num_kv_heads != 0) {
        reason = "PagedSparseAttention: key and value widths must be multiples of the number of KV heads.";
        return false;
    }
    config.k_head_size = key_width / config.num_kv_heads;
    config.v_head_size = value_width / config.num_kv_heads;
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
    // The page size is KEY_CACHE's dim 2; if it is dynamic the merge-unit/page fit can't be decided.
    // By-channel quantized key pages carry parameter rows in dim 2, so quantized caches use the derived page size.
    const auto page_size =
        m_quant.enabled() ? m_quant.block_size : getInputShapeAtPort(InputIdx::KEY_CACHE).getDims()[2];
    // Adjacent selected blocks of one page are attended as one run of up to a page of rows; with a dynamic page size
    // every block is its own run.
    m_max_run_rows = page_size != Shape::UNDEFINED_DIM ? page_size : m_config.key_block_size;
#ifdef OPENVINO_ARCH_X86_64
    // AMX matrix products over the selected rows of a work item (see sparse_brgemm_kernel.hpp), for the prefill and the
    // decode of a large selection: bf16 / f16 data on hardware with the matching AMX, a pack of several query heads
    // (the M of the products), and the causal selection.
    const auto query_prec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
    const bool amx =
        (query_prec == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16)) ||
        (query_prec == ov::element::f16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16));
    const size_t pack = ov::Extensions::Cpu::max_q_head_pack_size(m_config.num_heads,
                                                                  m_config.num_kv_heads,
                                                                  m_config.num_selection_heads);
    if (amx && m_config.causal && ov::Extensions::Cpu::SparseBrgemmKernels::supports_pack(pack)) {
        m_brgemm = get_sparse_brgemm_kernels(context->getParamsCache(),
                                             pack,
                                             Dh,
                                             m_config.v_head_size,
                                             ov::Extensions::Cpu::kGatherTileRows,
                                             query_prec);
    }
#endif
}

ov::Extensions::Cpu::SparsePAScratchLayout SparsePA::scratchLayout() const {
#ifdef OPENVINO_ARCH_X86_64
    const auto brgemm_sizes = ov::Extensions::Cpu::sparse_brgemm_sizes(m_brgemm.handle);
#else
    const ov::Extensions::Cpu::SparseBrgemmSizes brgemm_sizes{};
#endif
    return {m_config.k_head_size,
            m_config.v_head_size,
            m_max_run_rows,
            ov::Extensions::Cpu::max_q_head_pack_size(m_config.num_heads,
                                                      m_config.num_kv_heads,
                                                      m_config.num_selection_heads),
            getOriginalInputPrecisionAtPort(InputIdx::QUERY).size(),
            m_quant,
            brgemm_sizes};
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
    // whole history each call. The page table and selection inputs are i32; SCALE keeps its own precision and is read
    // as f32. Plain caches: the query type, or f16 next to f32 data like PagedAttention.
    if (!m_quant.enabled()) {
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

    // The width of selected_block_indices is only needed by phase 2 (it sizes the AMX run decomposition), so it need
    // not be static.
    const auto layout = scratchLayout();
    // Phase 2 scratch: one block per worker thread, then one token id per token for the kernel's work list (see
    // WorkList).
    const size_t thread_scratch_bytes =
        static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()) * layout.total;
    const size_t attention_bytes = thread_scratch_bytes + query.size(0) * sizeof(int32_t);
    // Phase 1 scratch (the slot mapping and the by-channel requantization buffer). The phases are separated by a
    // barrier, so they share the start of one scratchpad request: a second request would free the first one's memory.
    const size_t block_size = m_quant.enabled() ? m_quant.block_size : key_cache.size(2);
    const size_t write_bytes = writeKVScratchBytes(query.size(0), block_size);
    const auto scratch = context->getScratchPad()->createScratchPadMem(
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8,
                                               ov::intel_cpu::Shape{std::max(attention_bytes, write_bytes)}));

    // Phase 1: write the current K/V into the caches (parallel over token x kv head); it completes before phase 2,
    // the sparse attention, reads them.
    writeKV(scratch->getDataAs<uint8_t>());
    // Phase 2: sparse attention, parallel over (query token, query head pack), with per-thread scratch that the kernel
    // carves with the same layout.
    ov::Extensions::Cpu::XARCH::sparse_pa_compute(
        query,
        key_cache,
        value_cache,
        block_indices,
        block_indices_begins,
        subsequence_begins,
        past_lens,
        sel_indices,
        sel_count,
        out,
        c.num_heads,
        c.num_kv_heads,
        c.k_head_size,
        c.v_head_size,
        c.key_block_size,
        c.num_selection_heads,
        c.causal,
        scale,
        m_quant,
        m_max_run_rows,
#ifdef OPENVINO_ARCH_X86_64
        m_brgemm.handle,
#else
        ov::Extensions::Cpu::SparseBrgemm{},
#endif
        scratch->getDataAs<uint8_t>(),
        layout.total,
        reinterpret_cast<int32_t*>(scratch->getDataAs<uint8_t>() + thread_scratch_bytes),
        context->getCpuParallel());
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
        // The caches are either both plain f32/f16/bf16 or both quantized to u8/u4 (see SparsePAQuant).
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
