// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.h"

#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/attn_memcpy.hpp"
#include "nodes/kernels/scaled_attn/attn_quant.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_pa_kernel.hpp"
#include "nodes/paged_attn.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include <cpu/x64/cpu_isa_traits.hpp>
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
    OPENVINO_ASSERT(deriveConfig(*sparse_pa, m_config, errorMessage), errorMessage);
    if (!deriveQuant(errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
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

void SparsePA::writeKV() {
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
    m_slot_mapping.resize<int32_t>({tokens});
    size_t idx = 0;
    for (size_t seq = 0; seq < past_lens.size(0); seq++) {
        const int32_t q_len = subsequence_begins.ptr<int32_t>()[seq + 1] - subsequence_begins.ptr<int32_t>()[seq];
        const int32_t past = past_lens.ptr<int32_t>()[seq];
        const int32_t first_block = block_indices_begins.ptr<int32_t>()[seq];
        for (int32_t j = 0; j < q_len; j++) {
            const auto pos = static_cast<size_t>(past + j);
            const auto block = static_cast<size_t>(block_indices.ptr<int32_t>()[first_block + pos / block_size]);
            m_slot_mapping.ptr<int32_t>()[idx++] = static_cast<int32_t>(block * block_size + pos % block_size);
        }
    }

    const auto key_4d = key.reshape({tokens, c.num_kv_heads, 1, c.k_head_size});
    const auto value_4d = value.reshape({tokens, c.num_kv_heads, 1, c.v_head_size});
    if (!m_quant.enabled()) {
        ov::Extensions::Cpu::XARCH::paged_attn_memcpy(key_4d,
                                                      value_4d,
                                                      key_cache,
                                                      value_cache,
                                                      m_slot_mapping,
                                                      context->getCpuParallel());
        return;
    }
    ov::Extensions::Cpu::QuantizeParam quant_param{m_quant.key_bychannel,
                                                   false,
                                                   false,
                                                   m_quant.key_group_size,
                                                   m_quant.value_group_size};
    if (m_quant.key_bychannel) {
        m_temp_buffer.resize<float>({static_cast<size_t>(parallel_get_max_threads()),
                                     block_size,
                                     c.num_heads,
                                     std::max(c.k_head_size, c.v_head_size)});
    }
    ov::Extensions::Cpu::XARCH::paged_attn_quantkv(key_4d,
                                                   value_4d,
                                                   key_cache,
                                                   value_cache,
                                                   past_lens,
                                                   subsequence_begins,
                                                   block_indices,
                                                   block_indices_begins,
                                                   m_slot_mapping,
                                                   m_temp_buffer,
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
        !key_cache_ps[1].is_static() || !selected_ps[1].is_static() || !selected_ps[2].is_static()) {
        reason = "PagedSparseAttention CPU implementation requires static head dimensions.";
        return false;
    }
    config.num_kv_heads = static_cast<size_t>(key_cache_ps[1].get_length());
    config.num_selection_heads = static_cast<size_t>(selected_ps[1].get_length());
    config.k_head_size = static_cast<size_t>(key_ps[1].get_length()) / config.num_kv_heads;
    config.v_head_size = static_cast<size_t>(value_ps[1].get_length()) / config.num_kv_heads;
    config.num_heads = static_cast<size_t>(query_ps[1].get_length()) / config.k_head_size;
    config.key_block_size = static_cast<size_t>(op.get_key_block_size());
    config.max_selected_blocks = static_cast<size_t>(selected_ps[2].get_length());
    config.causal = true;
    if (config.num_selection_heads != 1 && config.num_selection_heads != config.num_kv_heads &&
        config.num_selection_heads != config.num_heads) {
        reason = "PagedSparseAttention CPU implementation requires 1, num_kv_heads, or num_heads selection heads.";
        return false;
    }
    return true;
}

void SparsePA::createPrimitive() {
    const auto Dh = m_config.k_head_size;
    // The page size is KEY_CACHE's dim 2; if it is dynamic the merge-unit/page fit can't be decided.
    // By-channel quantized key pages carry parameter rows in dim 2, so quantized caches use the derived page size.
    const auto page_size =
        m_quant.enabled() ? m_quant.block_size : getInputShapeAtPort(InputIdx::KEY_CACHE).getDims()[2];
    // Adjacent selected blocks of one page are attended as one run of up to a page of rows; with a dynamic page size
    // every block is its own run.
    m_max_run_rows = page_size != Shape::UNDEFINED_DIM ? page_size : m_config.key_block_size;
#ifdef OPENVINO_ARCH_X86_64
    const auto Bk = m_config.key_block_size;
    // JIT kernel limits: head_size must be a whole number of 32-wide bf16/f16 B-tiles, up to kMaxHeadSize (sizes
    // above 192 take the kernel's K-chunk loop). The JIT kernel's own block_size (rows per AMX call) must be a multiple
    // of 16, but Bk itself need not be -- see m_amx_unit_blocks below, which merges adjacent selected blocks up to a
    // valid 16-row-multiple chunk.
    if (Dh % 32 == 0 && Dh <= ov::Extensions::Cpu::JitMatMulVecAMX::kMaxHeadSize && Bk > 0 &&
        page_size != Shape::UNDEFINED_DIM) {
        const auto qprec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        // The AMX path reads exclusively from KEY_CACHE/VALUE_CACHE (never the transient
        // current-step KEY/VALUE inputs), so those are the precisions that must match QUERY and
        // the target ISA, not KEY/VALUE's.
        const auto kprec = getOriginalInputPrecisionAtPort(InputIdx::KEY_CACHE);
        const auto vprec = getOriginalInputPrecisionAtPort(InputIdx::VALUE_CACHE);
        // Quantized caches are dequantized into AMX-typed scratch chunk by chunk, so only plain caches need to
        // match the query precision.
        const bool caches_match = m_quant.enabled() || (qprec == kprec && qprec == vprec);
        if (caches_match && qprec == ov::element::bf16 &&
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16)) {
            m_amx_prec = ov::element::bf16;
        } else if (caches_match && qprec == ov::element::f16 &&
                   dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16)) {
            m_amx_prec = ov::element::f16;
        }
    }
    if (m_amx_prec != ov::element::dynamic) {
        // Smallest number of adjacent Bk-sized blocks whose combined row count is a multiple of 16
        // (e.g. Bk=4 -> 4 blocks = 16 rows; Bk=16 -> 1 block = 16 rows; Bk=2 -> 8 blocks = 16 rows).
        m_amx_unit_blocks = 16 / std::gcd(static_cast<size_t>(16), Bk);
        const auto unit_rows = m_amx_unit_blocks * Bk;
        // Unlike sparse_sdpa's dense cache (where a merge run is only bounded by S), a merge unit
        // here must also fit inside one physical page -- if the page itself can't host even one
        // full unit, the AMX kernels would never be invocable, so skip building them entirely.
        if (unit_rows <= page_size) {
            m_gemv_qk1 = std::make_shared<ov::Extensions::Cpu::JitMatMulVecAMX>(static_cast<int>(Dh),
                                                                                static_cast<int>(unit_rows),
                                                                                m_amx_prec);
            m_gemv_qk4 = std::make_shared<ov::Extensions::Cpu::JitMatMulVecAMX>(static_cast<int>(Dh),
                                                                                static_cast<int>(4 * unit_rows),
                                                                                m_amx_prec);
        } else {
            m_amx_prec = ov::element::dynamic;
            m_amx_unit_blocks = 0;
        }
    }
#endif
    // Per-worker-thread scratch of the kernel, so nothing is allocated while computing.
#ifdef OPENVINO_ARCH_X86_64
    const size_t amx_unit_blocks = m_amx_prec != ov::element::dynamic ? m_amx_unit_blocks : 0;
#else
    const size_t amx_unit_blocks = 0;
#endif
#ifdef OPENVINO_ARCH_X86_64
    // Prefill with AMX matrix products over the whole selection of a work item (see attend_rows_brgemm in the kernel):
    // bf16 / f16 data on hardware with the matching AMX, a pack of several query heads (the M of the products; the
    // kernel runs them as an M tail, so fewer than 32), and the causal selection.
    {
        const auto query_prec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        const bool amx =
            (query_prec == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16)) ||
            (query_prec == ov::element::f16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16));
        const size_t pack = ov::Extensions::Cpu::max_q_head_pack_size(m_config.num_heads,
                                                                      m_config.num_kv_heads,
                                                                      m_config.num_selection_heads);
        if (amx && m_config.causal && pack > 1 && pack < ov::intel_cpu::BrgemmKernel::get_mblk_size()) {
            constexpr size_t chunk_rows = 32;  // kBrgemmChunkRows in the kernel
            // Rows of one item: the selected blocks and the causal tail (fewer than key_block_size rows).
            const size_t stride =
                ov::intel_cpu::rnd_up(m_config.max_selected_blocks * m_config.key_block_size + m_config.key_block_size,
                                      chunk_rows);
            const auto Dv = m_config.v_head_size;
            m_brgemm_qk =
                std::make_shared<ov::intel_cpu::BrgemmKernel>(pack, chunk_rows, Dh, Dh, Dh, stride, true, query_prec);
            // The weights are stored in the data type over the f32 score rows, so the row stride doubles.
            m_brgemm_pv = std::make_shared<ov::intel_cpu::BrgemmKernel>(pack,
                                                                        Dv,
                                                                        chunk_rows,
                                                                        2 * stride,
                                                                        Dv,
                                                                        Dv,
                                                                        false,
                                                                        query_prec);
            m_brgemm_pv_acc = std::make_shared<ov::intel_cpu::BrgemmKernel>(pack,
                                                                            Dv,
                                                                            chunk_rows,
                                                                            2 * stride,
                                                                            Dv,
                                                                            Dv,
                                                                            false,
                                                                            query_prec,
                                                                            true);
            m_brgemm_handles = {m_brgemm_qk.get(), m_brgemm_pv.get(), m_brgemm_pv_acc.get(), stride};
        }
    }
#endif
    const ov::Extensions::Cpu::SparsePAScratchLayout layout(
        Dh,
        m_config.v_head_size,
        m_config.key_block_size,
        m_max_run_rows,
        m_config.max_selected_blocks,
        amx_unit_blocks,
        ov::Extensions::Cpu::max_q_head_pack_size(m_config.num_heads,
                                                  m_config.num_kv_heads,
                                                  m_config.num_selection_heads),
        getOriginalInputPrecisionAtPort(InputIdx::QUERY).size(),
        m_quant,
        ov::Extensions::Cpu::sparse_pa_brgemm_sizes(m_brgemm_handles));
    m_scratch_bytes_per_thread = layout.total;
    const auto num_worker_threads = context->getCpuParallel()->get_num_worker_threads();
    m_scratch = context->getScratchPad()->createScratchPadMem(std::make_shared<CpuBlockedMemoryDesc>(
        ov::element::u8,
        ov::intel_cpu::Shape{static_cast<size_t>(num_worker_threads), m_scratch_bytes_per_thread}));
}

void SparsePA::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // QUERY / KEY / VALUE / KEY_CACHE / VALUE_CACHE are echoed at whatever precision (f32/f16/
    // bf16) was negotiated upstream -- the kernel reads/writes all of them through precision
    // conversion in sparse_pa_compute() rather than hardcoding f32, since the paged caches grow
    // every step and forcing a whole-tensor Reorder on every call would redo that conversion for
    // the entire history each time. The page-table and selection-plan inputs stay i32; SCALE keeps its own precision
    // (the op requires it to match query/key/value, which dynamic-shape inference re-validates) and is read as f32.
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

    // Phase 1: write the current K/V into the caches (parallel over token x kv head); it completes before phase 2,
    // the sparse attention, reads them.
    writeKV();
    // Phase 2: sparse attention, parallel over (query token, query head), with per-thread scratch.
    ov::Extensions::Cpu::XARCH::sparse_pa_compute(query,
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
#ifdef OPENVINO_ARCH_X86_64
                                                  m_amx_prec,
                                                  m_gemv_qk1.get(),
                                                  m_gemv_qk4.get(),
                                                  m_amx_unit_blocks,
#else
                                                  ov::element::dynamic,
                                                  nullptr,
                                                  nullptr,
                                                  0,
#endif
                                                  m_max_run_rows,
                                                  m_brgemm_handles,
                                                  m_scratch->getDataAs<uint8_t>(),
                                                  m_scratch_bytes_per_thread,
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
        const auto query_prec = op->get_input_element_type(InputIdx::QUERY);
        if (is_plain(key_cache_prec) && (key_cache_prec != query_prec || value_cache_prec != query_prec)) {
            errorMessage = "SparsePA CPU implementation requires plain caches to have the query precision.";
            return false;
        }
        if (!(is_plain(key_cache_prec) && is_plain(value_cache_prec)) &&
            !(is_quantized(key_cache_prec) && is_quantized(value_cache_prec))) {
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
