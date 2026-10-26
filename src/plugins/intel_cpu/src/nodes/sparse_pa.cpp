// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.h"

#include <cstddef>
#include <numeric>
#include <vector>

#include "cpu_memory.h"
#include "cpu_parallel.hpp"
#include "cpu_shape.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/sparse_pa_kernel.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "transformations/cpu_opset/x64/op/sparse_pa.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif

namespace ov::intel_cpu::node {

namespace {

using InputIdx = ov::op::internal::SparsePA::InputIdx;

}  // namespace

SparsePA::SparsePA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto sparse_pa = ov::as_type_ptr<const ov::op::internal::SparsePA>(op);
    OPENVINO_ASSERT(sparse_pa, "Expected SparsePA node");
    m_config = sparse_pa->get_config();
}

void SparsePA::createPrimitive() {
    const auto Dh = m_config.k_head_size;
#ifdef OPENVINO_ARCH_X86_64
    const auto Bk = m_config.key_block_size;
    // The page size is KEY_CACHE's dim 2; if it is dynamic the merge-unit/page fit can't be decided.
    const auto page_size = getInputShapeAtPort(InputIdx::KEY_CACHE).getDims()[2];
    // JIT kernel limits: head_size must be a whole number of 32-wide bf16/f16 B-tiles, capped at 6
    // tiles. The JIT kernel's own block_size (rows per AMX call) must be a multiple of 16, but Bk
    // itself need not be -- see m_amx_unit_blocks below, which merges adjacent selected blocks up
    // to a valid 16-row-multiple chunk.
    if (Dh % 32 == 0 && Dh <= 192 && Bk > 0 && page_size != Shape::UNDEFINED_DIM) {
        const auto qprec = getOriginalInputPrecisionAtPort(InputIdx::QUERY);
        // The AMX path reads exclusively from KEY_CACHE/VALUE_CACHE (never the transient
        // current-step KEY/VALUE inputs), so those are the precisions that must match QUERY and
        // the target ISA, not KEY/VALUE's.
        const auto kprec = getOriginalInputPrecisionAtPort(InputIdx::KEY_CACHE);
        const auto vprec = getOriginalInputPrecisionAtPort(InputIdx::VALUE_CACHE);
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
}

void SparsePA::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    // QUERY / KEY / VALUE / KEY_CACHE / VALUE_CACHE are echoed at whatever precision (f32/f16/
    // bf16) was negotiated upstream -- the kernel reads/writes all of them through precision
    // conversion in sparse_pa_compute() rather than hardcoding f32, since the paged caches grow
    // every step and forcing a whole-tensor Reorder on every call would redo that conversion for
    // the entire history each time. The page-table and selection-plan inputs stay i32; SCALE is read as f32.
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        const bool is_i32_port = i == InputIdx::BLOCK_INDICES || i == InputIdx::BLOCK_INDICES_BEGINS ||
                                 i == InputIdx::SUBSEQUENCE_BEGINS || i == InputIdx::PAST_LENS ||
                                 i == InputIdx::SELECTED_BLOCK_INDICES || i == InputIdx::SELECTED_BLOCK_COUNT ||
                                 i == InputIdx::QB_START;
        const bool is_f32_port = i == InputIdx::SCALE;
        const ov::element::Type precision = is_i32_port   ? ov::element::i32
                                            : is_f32_port ? ov::element::f32
                                                          : getOriginalInputPrecisionAtPort(i);
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
    // QB_START is optional: absent means one query block per token, i.e. subsequence_begins.
    PlainTensor qb_start(getParentEdges().size() > InputIdx::QB_START
                             ? getSrcMemoryAtPort(InputIdx::QB_START)
                             : getSrcMemoryAtPort(InputIdx::SUBSEQUENCE_BEGINS));
    const float scale = *getSrcDataAtPortAs<const float>(InputIdx::SCALE);
    PlainTensor out(getDstMemoryAtPort(0));

    ov::Extensions::Cpu::XARCH::sparse_pa_compute(query,
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
                                                  qb_start,
                                                  out,
                                                  c.num_heads,
                                                  c.num_kv_heads,
                                                  c.k_head_size,
                                                  c.v_head_size,
                                                  c.key_block_size,
                                                  c.num_selection_heads,
                                                  c.causal,
                                                  scale,
#ifdef OPENVINO_ARCH_X86_64
                                                  m_amx_prec,
                                                  m_gemv_qk1.get(),
                                                  m_gemv_qk4.get(),
                                                  m_amx_unit_blocks,
#endif
                                                  context->getCpuParallel());
}

bool SparsePA::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto sparse_pa = ov::as_type_ptr<const ov::op::internal::SparsePA>(op);
        if (!sparse_pa) {
            errorMessage = "Only SparsePA operation is supported";
            return false;
        }
        static constexpr size_t float_ports[] = {InputIdx::QUERY,
                                                 InputIdx::KEY,
                                                 InputIdx::VALUE,
                                                 InputIdx::KEY_CACHE,
                                                 InputIdx::VALUE_CACHE};
        for (auto port : float_ports) {
            const auto prec = op->get_input_element_type(port);
            if (prec != ov::element::f32 && prec != ov::element::f16 && prec != ov::element::bf16) {
                errorMessage = "SparsePA CPU implementation only supports f32, f16, or bf16 data inputs.";
                return false;
            }
        }
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
        if (op->get_input_size() > InputIdx::QB_START &&
            op->get_input_element_type(InputIdx::QB_START) != ov::element::i32) {
            errorMessage = "SparsePA CPU implementation requires i32 index inputs.";
            return false;
        }
        // Sinks are not supported yet: SINKS must be an empty tensor.
        const auto& sinks_ps = op->get_input_partial_shape(InputIdx::SINKS);
        if (!sinks_ps.is_static() || ov::shape_size(sinks_ps.to_shape()) != 0) {
            errorMessage = "SparsePA CPU implementation does not support sinks.";
            return false;
        }
        // Only the indices/count plan form is implemented: a non-empty SELECTED_BLOCK_MASK is not.
        if (op->get_input_size() > InputIdx::SELECTED_BLOCK_MASK) {
            const auto& mask_ps = op->get_input_partial_shape(InputIdx::SELECTED_BLOCK_MASK);
            if (!mask_ps.is_static() || ov::shape_size(mask_ps.to_shape()) != 0) {
                errorMessage = "SparsePA CPU implementation does not support selected_block_mask.";
                return false;
            }
        }
        std::string reason;
        if (!ov::op::internal::SparsePA::validate_config(sparse_pa->get_config(), reason)) {
            errorMessage = "SparsePA configuration is invalid: " + reason;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node