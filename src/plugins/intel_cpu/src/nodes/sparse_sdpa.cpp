// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "config.h"
#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_state.h"
#include "node.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/kernels/scaled_attn/attn_memcpy.hpp"
#include "nodes/kernels/scaled_attn/attn_quant.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_sdpa_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "transformations/cpu_opset/common/op/sparse_attention_with_kv_cache.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif

namespace ov::intel_cpu::node {

namespace {

using SparseAttentionOp = ov::op::v17::SparseAttention;
using SparseAttentionKVOp = ov::intel_cpu::SparseAttentionWithKVCache;

// Input order of ov::op::v17::SparseAttention.
struct InputIdx {
    static constexpr size_t QUERY = 0;        // [N, H, L, E]
    static constexpr size_t KEY = 1;          // [N, Hkv, S, E]
    static constexpr size_t VALUE = 2;        // [N, Hkv, S, Ev]
    static constexpr size_t SEL_INDICES = 3;  // block_indices [N, Hs, L, KB]
    static constexpr size_t SEL_COUNT = 4;    // block_count [N, Hs, L]
    static constexpr size_t KEY_LENGTH = 5;   // optional [N] or [1]
    static constexpr size_t COUNT_MIN = 5;
};

// BHLS KV cache descriptor with the logical length L; the capacity cap_L >= L along L is carried by the strides.
std::shared_ptr<CpuBlockedMemoryDesc> make_bhls_cache_desc(ov::element::Type prec,
                                                           size_t B,
                                                           size_t H,
                                                           size_t L,
                                                           size_t S,
                                                           size_t cap_L) {
    const VectorDims dims{B, H, L, S};
    return std::make_shared<CpuBlockedMemoryDesc>(prec,
                                                  Shape(dims),
                                                  dims,
                                                  VectorDims{0, 1, 2, 3},
                                                  0,
                                                  VectorDims{},
                                                  VectorDims{H * cap_L * S, cap_L * S, S, 1});
}

}  // namespace

SparseSDPA::SparseSDPA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (const auto kv_op = ov::as_type_ptr<const SparseAttentionKVOp>(op)) {
        m_fuse_concat = true;
        m_base_inputs = op->get_input_size() - 3;
        m_cache_prec = op->get_input_element_type(InputIdx::QUERY);
        // a side is compressed when the hint asks for u8 / u4 and the head size is static; the keys are quantized
        // by channel when that mode is requested (u8 only). Everything else (by-channel u4, TurboQuant, f16 / i8
        // hints) leaves the side a plain cache
        const auto& cpu_config = context->getConfig();
        auto make_spec = [&](ov::element::Type hint,
                             size_t group_size,
                             const ov::Dimension& head_dim,
                             bool by_channel,
                             ov::internal::CacheQuantAlgorithm alg) {
            ov::Extensions::Cpu::CacheSpec spec;
            spec.alg = ov::internal::CacheQuantAlgorithm::SCALAR;
            spec.by_channel = false;
            spec.precision = m_cache_prec;
            spec.group_size = head_dim.is_static() ? static_cast<size_t>(head_dim.get_length()) : 0;
            if (none_of(hint, ov::element::u8, ov::element::u4) || spec.group_size == 0 ||
                alg == ov::internal::CacheQuantAlgorithm::TURBO) {
                return spec;
            }
            const size_t head = spec.group_size;
            // same rule as SDPA: the group is the whole head unless the configured size divides it; for by-channel
            // quantization the group is a number of tokens
            const size_t group = (group_size == 0 || head % group_size != 0) ? head : group_size;
            if (by_channel) {
                if (hint != ov::element::u8) {
                    return spec;
                }
                spec.by_channel = true;
            } else if (hint == ov::element::u4 && group % 2 != 0) {
                return spec;
            }
            spec.precision = hint;
            spec.group_size = group;
            return spec;
        };
        m_key_spec = make_spec(cpu_config.keyCachePrecision,
                               cpu_config.keyCacheGroupSize,
                               op->get_input_partial_shape(InputIdx::KEY)[3],
                               cpu_config.keyCacheQuantMode == ov::intel_cpu::Config::CacheQuantMode::BY_CHANNEL,
                               cpu_config.keyCacheQuantAlg);
        m_value_spec = make_spec(cpu_config.valueCachePrecision,
                                 cpu_config.valueCacheGroupSize,
                                 op->get_input_partial_shape(InputIdx::VALUE)[3],
                                 false,
                                 cpu_config.valueCacheQuantAlg);
        m_key_block_size = static_cast<size_t>(kv_op->get_config().key_block_size);
        m_causal = kv_op->get_config().causal;
    } else {
        const auto sparse_attention = ov::as_type_ptr<const SparseAttentionOp>(op);
        OPENVINO_ASSERT(sparse_attention, "Expected SparseAttention node");
        m_base_inputs = op->get_input_size();
        m_key_block_size = static_cast<size_t>(sparse_attention->get_key_block_size());
        m_causal = sparse_attention->get_causal();
    }
    m_scale = 1.0F / std::sqrt(static_cast<float>(op->get_input_partial_shape(InputIdx::QUERY)[3].get_length()));
}

void SparseSDPA::createPrimitive() {
    const auto Dh = static_cast<size_t>(getInputShapeAtPort(InputIdx::QUERY).getDims()[3]);
#ifdef OPENVINO_ARCH_X86_64
    const auto Bk = m_key_block_size;
    // JIT kernel limits: head_size must be a whole number of 32-wide bf16/f16 B-tiles, up to kMaxHeadSize (sizes
    // above 192 take the kernel's K-chunk loop). The JIT kernel's own block_size (rows per AMX call) must be a multiple
    // of 16, but Bk itself need not be -- see m_amx_unit_blocks below, which merges adjacent selected blocks up to a
    // valid 16-row-multiple chunk.
    if (Dh % 32 == 0 && Dh <= ov::Extensions::Cpu::JitMatMulVecAMX::kMaxHeadSize && Bk > 0) {
        const auto qprec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        // the cache of the stateful form is always kept in the query's precision
        const auto kprec = m_fuse_concat ? m_cache_prec : getOriginalInputPrecisionAtPort(InputIdx::KEY);
        // VALUE must match too: the AMX chunk path's P@V step reads VALUE directly at this
        // precision via attn_acc_value_block() (zero-copy, same as the QK^T read of KEY) rather
        // than through the precision-agnostic load_row_f32() the scalar fallback uses.
        const auto vprec = m_fuse_concat ? m_cache_prec : getOriginalInputPrecisionAtPort(InputIdx::VALUE);
        if (qprec == kprec && qprec == vprec && qprec == ov::element::bf16 &&
            dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16)) {
            m_amx_prec = ov::element::bf16;
        } else if (qprec == kprec && qprec == vprec && qprec == ov::element::f16 &&
                   dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16)) {
            m_amx_prec = ov::element::f16;
        }
    }
    if (m_amx_prec != ov::element::dynamic) {
        // Smallest number of adjacent Bk-sized blocks whose combined row count is a multiple of 16
        // (e.g. Bk=4 -> 4 blocks = 16 rows; Bk=16 -> 1 block = 16 rows; Bk=2 -> 8 blocks = 16 rows).
        m_amx_unit_blocks = 16 / std::gcd(static_cast<size_t>(16), Bk);
        const auto unit_rows = m_amx_unit_blocks * Bk;
        m_gemv_qk1 = std::make_shared<ov::Extensions::Cpu::JitMatMulVecAMX>(static_cast<int>(Dh),
                                                                             static_cast<int>(unit_rows),
                                                                             m_amx_prec);
        m_gemv_qk4 = std::make_shared<ov::Extensions::Cpu::JitMatMulVecAMX>(static_cast<int>(Dh),
                                                                             static_cast<int>(4 * unit_rows),
                                                                             m_amx_prec);
    }
#endif
}

void SparseSDPA::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // QUERY / KEY / VALUE are echoed at whatever precision (f32/f16/bf16) was negotiated
    // upstream -- the reference kernel reads them through load_row_f32() rather than hardcoding
    // f32, since the dense caches grow every step and forcing a whole-tensor Reorder on every call
    // would redo that conversion for the entire history each time. SEL_INDICES / SEL_COUNT are
    // always i32 (enforced by the op's shape inference).
    if (m_fuse_concat) {
        // Same layout as ScaledDotProductAttention with fuse_concat: the state inputs keep the precision of the
        // MemoryInput nodes feeding them (no reorders, they are only proxies of the state), the cache outputs carry
        // the shapes only and are consumed by MemoryOutputStub.
        std::vector<PortConfigurator> stateInConfigs;
        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            ov::element::Type precision;
            if (i == InputIdx::SEL_INDICES || i == InputIdx::SEL_COUNT || i == InputIdx::KEY_LENGTH ||
                i == m_base_inputs) {
                precision = ov::element::i32;  // the index ports and beam_idx
            } else if (i > m_base_inputs) {
                precision = getParentEdgeAt(i)->getParent()->getOriginalOutputPrecisionAtPort(0);
            } else if (i == InputIdx::KEY || i == InputIdx::VALUE) {
                precision = m_cache_prec;
            } else {
                precision = getOriginalInputPrecisionAtPort(i);
            }
            stateInConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
        }
        const auto past_k_prec = getParentEdgeAt(m_base_inputs + 1)->getParent()->getOriginalOutputPrecisionAtPort(0);
        const auto past_v_prec = getParentEdgeAt(m_base_inputs + 2)->getParent()->getOriginalOutputPrecisionAtPort(0);
        std::vector<PortConfigurator> stateOutConfigs = {
            PortConfigurator{LayoutType::ncsp, getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0), false, -1},
            PortConfigurator{LayoutType::ncsp, past_k_prec, getOutputShapeAtPort(1), false, -1},
            PortConfigurator{LayoutType::ncsp, past_v_prec, getOutputShapeAtPort(2), false, -1}};
        addSupportedPrimDesc(stateInConfigs, stateOutConfigs, impl_desc_type::ref);
        return;
    }
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_index_port = i == InputIdx::SEL_INDICES || i == InputIdx::SEL_COUNT || i == InputIdx::KEY_LENGTH;
        const ov::element::Type precision = is_index_port ? ov::element::i32 : getOriginalInputPrecisionAtPort(i);
        inPortConfigs.emplace_back(LayoutType::ncsp, precision, getInputShapeAtPort(i), false, -1);
    }
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, getOriginalOutputPrecisionAtPort(0), getOutputShapeAtPort(0), false, -1}};
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void SparseSDPA::execute([[maybe_unused]] const dnnl::stream& strm) {
    PlainTensor query(getSrcMemoryAtPort(InputIdx::QUERY));
    PlainTensor key;
    PlainTensor value;
    ov::Extensions::Cpu::SparseKVQuant key_quant;
    ov::Extensions::Cpu::SparseKVQuant value_quant;
    if (m_fuse_concat) {
        updateKVCache();
        key.reset(m_k_state->internal_state_mem());
        value.reset(m_v_state->internal_state_mem());
        if (isQuantizedCache(m_key_spec)) {
            key_quant = {m_key_spec.precision,
                         m_key_spec.group_size,
                         m_key_spec.by_channel,
                         &m_k_state->get_scale_zp()};
        }
        if (isQuantizedCache(m_value_spec)) {
            value_quant = {m_value_spec.precision,
                           m_value_spec.group_size,
                           m_value_spec.by_channel,
                           &m_v_state->get_scale_zp()};
        }
    } else {
        key.reset(getSrcMemoryAtPort(InputIdx::KEY));
        value.reset(getSrcMemoryAtPort(InputIdx::VALUE));
    }
    PlainTensor sel_indices(getSrcMemoryAtPort(InputIdx::SEL_INDICES));
    PlainTensor sel_count(getSrcMemoryAtPort(InputIdx::SEL_COUNT));
    PlainTensor out(getDstMemoryAtPort(0));

    const size_t Hq = query.size(1);
    const size_t Hkv = key.size(1);
    const size_t Dh = query.size(3);
    const size_t Dv = value.size(3);
    const size_t Bk = m_key_block_size;
    const size_t H_sel = sel_indices.size(1);
    const bool causal = m_causal;

    const bool has_key_length = m_base_inputs > InputIdx::KEY_LENGTH;
    PlainTensor key_length;
    const int32_t* key_length_ptr = nullptr;
    size_t key_length_batch_stride = 0;
    if (has_key_length) {
        key_length.reset(getSrcMemoryAtPort(InputIdx::KEY_LENGTH));
        key_length_ptr = key_length.ptr<int32_t>(0);
        key_length_batch_stride = key_length.size(0) == 1 ? 0 : 1;
    }

    // Per-worker-thread f32 scratch row: q_buf[Dh] | k_buf[Dh] | acc[Dv] | v_buf[Dv]
    // (| raw[amx_raw_floats] when AMX-eligible). Per-worker-thread AMX run-decomposition scratch
    // (i64): chunk_block_start[K_max] | chunk_n_units[K_max] | leftover_block_start[K_max], where
    // K_max is SEL_INDICES' runtime last dimension -- the upper bound on nblocks for every
    // (b, sh, l) in this call, only known once actual input shapes are resolved (not in
    // createPrimitive()). Both regions are drawn from the plugin's DnnlScratchPad, which is a
    // single shared, grow-only buffer per engine: a second createScratchPadMem() call that grows
    // the pool frees the buffer backing any earlier call's pointer (see dnnl_scratch_pad.h /
    // MemoryBlockWithReuse::resize()). So this issues exactly ONE combined request per execute()
    // call -- sized for both regions together -- and slices it by byte offset, rather than
    // requesting the two regions separately. This scratch allocation needs GraphContext, so it
    // stays here rather than inside the cross-compiled sparse_sdpa_compute() kernel.
    size_t amx_raw_floats = 0;
    size_t runs_per_thread = 0;
#ifdef OPENVINO_ARCH_X86_64
    if (m_amx_prec != ov::element::dynamic) {
        amx_raw_floats = 4 * m_amx_unit_blocks * Bk;
        if (key_quant.enabled() || value_quant.enabled()) {
            // dequantized K and V rows of one AMX chunk (<= 4 merge units, bf16 / f16 elements), see
            // sparse_sdpa_compute()
            amx_raw_floats += (amx_raw_floats * (Dh + Dv) * 2 + sizeof(float) - 1) / sizeof(float);
        }
        const size_t K_max = sel_indices.size(3);
        runs_per_thread = 3 * K_max;
    }
#endif
    const size_t per_thread_floats = 2 * Dh + 2 * Dv + amx_raw_floats;
    const auto num_threads = static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads());
    const size_t float_bytes = num_threads * per_thread_floats * sizeof(float);
    const size_t float_bytes_aligned = (float_bytes + 7U) & ~size_t{7U};
    const size_t runs_bytes = num_threads * runs_per_thread * sizeof(int64_t);
    auto scratch_desc =
        std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, ov::intel_cpu::Shape{float_bytes_aligned + runs_bytes});
    const MemoryPtr scratch = context->getScratchPad()->createScratchPadMem(scratch_desc);
    uint8_t* const scratch_base_bytes = scratch->getDataAs<uint8_t>();
    float* const scratch_base = reinterpret_cast<float*>(scratch_base_bytes);
    int64_t* const runs_base =
        runs_per_thread > 0 ? reinterpret_cast<int64_t*>(scratch_base_bytes + float_bytes_aligned) : nullptr;

    ov::Extensions::Cpu::XARCH::sparse_sdpa_compute(query,
                                                    key,
                                                    value,
                                                    sel_indices,
                                                    sel_count,
                                                    out,
                                                    Hq,
                                                    Hkv,
                                                    Dh,
                                                    Dv,
                                                    Bk,
                                                    H_sel,
                                                    causal,
                                                    m_scale,
                                                    scratch_base,
                                                    per_thread_floats,
#ifdef OPENVINO_ARCH_X86_64
                                                    m_amx_prec,
                                                    m_gemv_qk1.get(),
                                                    m_gemv_qk4.get(),
                                                    m_amx_unit_blocks,
                                                    runs_base,
                                                    runs_per_thread,
#endif
                                                    key_length_ptr,
                                                    key_length_batch_stride,
                                                    key_quant,
                                                    value_quant,
                                                    context->getCpuParallel());
}

void SparseSDPA::assignState(const std::shared_ptr<VariableStateKVcache>& state, int idx) {
    CPU_NODE_ASSERT(m_fuse_concat, "has no KV cache states");
    const auto input_number = getOriginalInputsNumber();
    if (input_number - 2 == static_cast<size_t>(idx)) {
        m_k_state = state;
    } else if (input_number - 1 == static_cast<size_t>(idx)) {
        m_v_state = state;
    } else {
        CPU_NODE_THROW("Unexpected idx ",
                       idx,
                       " for a state in a node with type: ",
                       getTypeStr(),
                       " and name ",
                       getName());
    }
}

// Appends the current step's key/value to the cache states, in place whenever the capacity of the buffers allows it.
// The buffers are plain BHLS with a capacity along L carried by their strides (see make_bhls_cache_desc), so the rows
// of every (b, h) are contiguous along L, which is what the AMX path of the kernel relies on.
void SparseSDPA::updateKVCache() {
    CPU_NODE_ASSERT(m_k_state && m_v_state, "has null input states");
    const auto& cpu_parallel = context->getCpuParallel();

    PlainTensor cur_k(getSrcMemoryAtPort(InputIdx::KEY));
    PlainTensor cur_v(getSrcMemoryAtPort(InputIdx::VALUE));
    const size_t B = cur_k.size(0);
    const size_t H = cur_k.size(1);
    const size_t L1 = cur_k.size(2);
    CPU_NODE_ASSERT(cur_v.size(0) == B && cur_v.size(1) == H && cur_v.size(2) == L1,
                    "key and value of the current step have different shapes");

    const bool is_reset = m_k_state->is_reset_state();
    CPU_NODE_ASSERT(is_reset == m_v_state->is_reset_state(),
                    "key and value states are expected to be reset together, but only ",
                    (is_reset ? m_v_state->get_name() : m_k_state->get_name()),
                    " is not reset");

    // The state inputs are proxies of the states: their shapes tell the length (and batch) of the cache so far.
    const size_t beam_port = m_base_inputs;
    const size_t L0 = getParentEdgeAt(beam_port + 2)->getMemory().getStaticDims().at(2);
    CPU_NODE_ASSERT(B * (L0 + L1) > 0, "B or (L0+L1) is zero, B: ", B, ", L0: ", L0, ", L1: ", L1);
    const size_t L = L0 + L1;
    const bool has_past = !is_reset && L0 > 0;

    const auto beam_mem = getSrcMemoryAtPort(beam_port);
    CPU_NODE_ASSERT(beam_mem->getStaticDims().at(0) == B, "beam_idx size is not equal to the batch size: ", B);
    const auto* beam = beam_mem->getDataAs<const int32_t>();
    const size_t B_past = has_past ? getParentEdgeAt(beam_port + 2)->getMemory().getStaticDims().at(0) : B;
    // The beam table of the SDPA state is not used here: when the rows are not in the natural order
    // the cache is physically reordered (a rare path: beam search steps).
    bool need_gather = has_past && B_past != B;
    for (size_t b = 0; has_past && !need_gather && b < B; b++) {
        need_gather = beam[b] != static_cast<int32_t>(b);
    }
    if (need_gather) {
        for (size_t b = 0; b < B; b++) {
            CPU_NODE_ASSERT(beam[b] >= 0 && static_cast<size_t>(beam[b]) < B_past, "beam_idx is out of range");
        }
    }

    auto update = [&](const std::shared_ptr<VariableStateKVcache>& state,
                      const PlainTensor& cur,
                      size_t init_port,
                      const ov::Extensions::Cpu::CacheSpec& spec) {
        const auto prec = spec.precision;
        const bool quantized = isQuantizedCache(spec);
        const size_t S = cur.size(3);
        // {scale, zero point} of the groups, laid out as attn_quant_by_token() / attn_quant_by_channel() and the
        // dequantization of the kernel read it; it is not permuted along with the cache:
        //  - by token:   [capacity of L, B, H, 2 * S / group_size]
        //  - by channel: [2 * ceil(capacity of L / group_size), B, H, S]
        const bool by_channel = quantized && spec.by_channel;
        const size_t szp_cols = !quantized ? 0 : (by_channel ? S : 2 * S / spec.group_size);
        auto szp_rows = [&](size_t length) {
            return by_channel ? 2 * ((length + spec.group_size - 1) / spec.group_size) : length;
        };
        auto mem = state->internal_state_mem();

        // the buffer is reused when it is a BHLS buffer of the same B, H, S and has enough capacity
        size_t cap_L = 0;
        bool reusable = false;
        if (mem && !mem->getShape().hasZeroDims() && mem->getPrecision() == prec && !need_gather) {
            const auto& dims = mem->getStaticDims();
            const auto& strides = mem->getDescWithType<BlockedMemoryDesc>()->getStrides();
            if (dims.size() == 4 && dims[0] == B && dims[1] == H && dims[3] == S && strides[3] == 1 &&
                strides[2] == S && strides[1] % S == 0) {
                cap_L = strides[1] / S;
                reusable = cap_L >= L && (!has_past || dims[2] == L0);
                if (reusable && quantized) {
                    const auto& szp = state->get_scale_zp();
                    reusable = szp && szp.size(0) == szp_rows(cap_L) && szp.size(1) == B && szp.size(2) == H &&
                               szp.size(3) == szp_cols;
                }
            }
        }

        if (reusable) {
            mem->redefineDesc(make_bhls_cache_desc(prec, B, H, L, S, cap_L));
        } else {
            cap_L = 2 * L;
            auto new_mem = std::make_shared<Memory>(getEngine(), make_bhls_cache_desc(prec, B, H, cap_L, S, cap_L));
            PlainTensor new_szp;
            if (quantized) {
                new_szp.resize<float>({szp_rows(cap_L), B, H, szp_cols});
            }
            if (has_past) {
                CPU_NODE_ASSERT(mem && mem->getPrecision() == prec, "has no past state of the expected precision");
                PlainTensor old_cache(mem);
                PlainTensor new_cache(new_mem);
                CPU_NODE_ASSERT(old_cache.size(1) == H && old_cache.size(2) == L0 && old_cache.size(3) == S,
                                "past state shape does not match the current key/value");
                // rows of a (b, h) are contiguous along L; a u4 element takes half a byte
                const size_t row_bytes = L0 * S * prec.bitwidth() / 8;
                cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
                    const size_t b_src = need_gather ? static_cast<size_t>(beam[b]) : b;
                    std::memcpy(new_cache.ptr_v(b, h, 0), old_cache.ptr_v(b_src, h, 0), row_bytes);
                });
                if (quantized) {
                    const auto& old_szp = state->get_scale_zp();
                    const size_t past_rows = szp_rows(L0);
                    CPU_NODE_ASSERT(old_szp && old_szp.size(0) >= past_rows && old_szp.size(1) == B_past &&
                                        old_szp.size(2) == H && old_szp.size(3) == szp_cols,
                                    "past scale / zero point does not match the current key/value");
                    cpu_parallel->parallel_for3d(past_rows, B, H, [&](size_t pos, size_t b, size_t h) {
                        const size_t b_src = need_gather ? static_cast<size_t>(beam[b]) : b;
                        std::memcpy(new_szp.ptr<float>(pos, b, h),
                                    old_szp.ptr<float>(pos, b_src, h),
                                    szp_cols * sizeof(float));
                    });
                }
            }
            new_mem->redefineDesc(make_bhls_cache_desc(prec, B, H, L, S, cap_L));
            state->assign_internal_state(new_mem);
            if (quantized) {
                state->set_scale_zp(new_szp);
            }
            mem = new_mem;
        }
        state->assign_internal_state_max_size(B * H * cap_L * S);

        // by channel the quantization of the tokens already in the last (partial) group is redone together with
        // the new ones, see attn_quant_by_channel()
        auto quantize = [&](const PlainTensor& src, const PlainTensor& dst, const PlainTensor& szp, size_t from) {
            if (by_channel) {
                ov::Extensions::Cpu::XARCH::attn_quant_by_channel(src, dst, szp, from, spec.group_size, cpu_parallel);
            } else {
                ov::Extensions::Cpu::XARCH::attn_quant_by_token(src, dst, szp, from, spec.group_size, cpu_parallel);
            }
        };
        PlainTensor cache(mem);
        if (is_reset && L0 > 0) {
            // the state was reset (or never set) and has an initial value: it is provided by the state input
            PlainTensor init(getSrcMemoryAtPort(init_port));
            CPU_NODE_ASSERT(init.size(0) == B && init.size(1) == H && init.size(2) == L0 && init.size(3) == S,
                            "the initial state shape does not match the current key/value");
            if (quantized) {
                quantize(init, cache, state->get_scale_zp(), 0);
            } else {
                cpu_parallel->parallel_for2d(B, H, [&](size_t b, size_t h) {
                    cpu_convert(init.ptr_v(b, h, 0), cache.ptr_v(b, h, 0), init.get_precision(), prec, L0 * S);
                });
            }
        }
        if (quantized) {
            quantize(cur, cache, state->get_scale_zp(), L0);
        } else {
            ov::Extensions::Cpu::XARCH::attn_memcpy2d(cur, cache, L0, cpu_parallel);
        }
    };
    update(m_k_state, cur_k, beam_port + 1, m_key_spec);
    update(m_v_state, cur_v, beam_port + 2, m_value_spec);
}

bool SparseSDPA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (const auto kv_op = ov::as_type_ptr<const SparseAttentionKVOp>(op)) {
            const auto& cfg = kv_op->get_config();
            if (cfg.query_block_size != 1 || cfg.key_block_size <= 0) {
                errorMessage = "SparseAttentionWithKVCache CPU implementation only supports query_block_size == 1 and "
                               "a positive key_block_size.";
                return false;
            }
            const auto& q_ps = op->get_input_partial_shape(InputIdx::QUERY);
            if (q_ps.rank().is_dynamic() || q_ps.size() != 4 || q_ps[3].is_dynamic()) {
                errorMessage = "SparseAttentionWithKVCache CPU implementation requires a 4D query with a static head "
                               "size.";
                return false;
            }
            const auto q_prec = op->get_input_element_type(InputIdx::QUERY);
            if (none_of(q_prec, ov::element::f32, ov::element::f16, ov::element::bf16)) {
                errorMessage = "SparseAttentionWithKVCache CPU implementation only supports f32, f16, or bf16 query.";
                return false;
            }
            return true;
        }
        const auto sparse_attention = ov::as_type_ptr<const SparseAttentionOp>(op);
        if (!sparse_attention) {
            errorMessage = "Only SparseAttention or SparseAttentionWithKVCache operations are supported";
            return false;
        }
        if (op->get_input_size() != InputIdx::COUNT_MIN && op->get_input_size() != InputIdx::COUNT_MIN + 1) {
            errorMessage = "SparseAttention CPU implementation only supports the 5-input form or the 6-input form "
                           "(with key_length); the scale and sinks inputs are not supported.";
            return false;
        }
        if (sparse_attention->get_query_block_size() != 1) {
            errorMessage = "SparseAttention CPU implementation only supports query_block_size == 1.";
            return false;
        }
        if (sparse_attention->get_key_block_size() <= 0) {
            errorMessage = "SparseAttention key_block_size must be positive.";
            return false;
        }
        const auto& q_shape = op->get_input_partial_shape(InputIdx::QUERY);
        if (q_shape.rank().is_dynamic() || q_shape[3].is_dynamic()) {
            errorMessage = "SparseAttention CPU implementation requires a static query head size.";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::QUERY, InputIdx::KEY, InputIdx::VALUE};
        for (auto port : float_ports) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "SparseAttention CPU implementation only supports f32, f16, or bf16 data inputs.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node
