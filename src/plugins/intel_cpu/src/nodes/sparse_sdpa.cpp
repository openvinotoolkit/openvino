// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "cpu_shape.h"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_sdpa_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif

namespace ov::intel_cpu::node {

namespace {

using SparseAttentionOp = ov::op::v17::SparseAttention;

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

}  // namespace

SparseSDPA::SparseSDPA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto sparse_attention = ov::as_type_ptr<const SparseAttentionOp>(op);
    OPENVINO_ASSERT(sparse_attention, "Expected SparseAttention node");
    m_key_block_size = static_cast<size_t>(sparse_attention->get_key_block_size());
    m_causal = sparse_attention->get_causal();
    m_scale = 1.0F / std::sqrt(static_cast<float>(op->get_input_partial_shape(InputIdx::QUERY)[3].get_length()));
}

void SparseSDPA::createPrimitive() {
    const auto Dh = static_cast<size_t>(getInputShapeAtPort(InputIdx::QUERY).getDims()[3]);
#ifdef OPENVINO_ARCH_X86_64
    const auto Bk = m_key_block_size;
    // JIT kernel limits: head_size must be a whole number of 32-wide bf16/f16 B-tiles, capped at 6
    // tiles. The JIT kernel's own block_size (rows per AMX call) must be a multiple of 16, but Bk
    // itself need not be -- see m_amx_unit_blocks below, which merges adjacent selected blocks up
    // to a valid 16-row-multiple chunk.
    if (Dh % 32 == 0 && Dh <= 192 && Bk > 0) {
        const auto qprec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        const auto kprec = getOriginalInputPrecisionAtPort(InputIdx::KEY);
        // VALUE must match too: the AMX chunk path's P@V step reads VALUE directly at this
        // precision via attn_acc_value_block() (zero-copy, same as the QK^T read of KEY) rather
        // than through the precision-agnostic load_row_f32() the scalar fallback uses.
        const auto vprec = getOriginalInputPrecisionAtPort(InputIdx::VALUE);
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
    PlainTensor key(getSrcMemoryAtPort(InputIdx::KEY));
    PlainTensor value(getSrcMemoryAtPort(InputIdx::VALUE));
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

    const bool has_key_length = getParentEdges().size() > InputIdx::KEY_LENGTH;
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
                                                    context->getCpuParallel());
}

bool SparseSDPA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto sparse_attention = ov::as_type_ptr<const SparseAttentionOp>(op);
        if (!sparse_attention) {
            errorMessage = "Only SparseAttention operation is supported";
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
