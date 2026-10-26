// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/cpu_opset/x64/op/sparse_pa.hpp"

namespace ov::intel_cpu::node {

/// @brief F32 reference implementation of ov::op::internal::SparsePA (paged-cache sparse-
/// attention consumer for Qwen Sparse Attention).
///
/// Writes the current-step K/V into the caller-owned paged main KV cache addressed through
/// the shared PagedAttention page table, then performs the same scalar online-softmax
/// selected-block + causal-tail attention as the dense SparseSDPA node, with every K/V read
/// resolved through that same page table instead of a flat dense `[B, Hkv, S, D]` cache.
/// SCALE is a runtime input; sinks are not supported yet (SINKS must be an empty tensor) and an absent
/// QB_START means subsequence_begins. The optional SELECTED_BLOCK_MASK form is rejected.
/// Selected-block QK^T has a bf16/f16 AMX fast path mirroring SparseSDPA's, adapted for the
/// paged cache: merge runs are additionally capped at physical-page boundaries, since two
/// logically-adjacent selected blocks are only guaranteed memory-contiguous within the same
/// physical page.
class SparsePA : public Node {
public:
    SparsePA(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::SparsePA;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    // Only the token-major inputs gate execution: SINKS (and the optional SELECTED_BLOCK_MASK) is an
    // empty tensor when unused, which must not make the node skip.
    [[nodiscard]] bool neverExecute() const override {
        using InputIdx = ov::op::internal::SparsePA::InputIdx;
        const auto* pd = getSelectedPrimitiveDescriptor();
        return pd->hasZeroInputDimsAtPort(InputIdx::QUERY) || pd->hasZeroInputDimsAtPort(InputIdx::KEY) ||
               pd->hasZeroInputDimsAtPort(InputIdx::VALUE);
    }
    [[nodiscard]] bool isExecutable() const override {
        using InputIdx = ov::op::internal::SparsePA::InputIdx;
        return !isInputTensorAtPortEmpty(InputIdx::QUERY) && !isInputTensorAtPortEmpty(InputIdx::KEY) &&
               !isInputTensorAtPortEmpty(InputIdx::VALUE);
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    ov::op::internal::SparsePA::Config m_config{};

    // AMX QK^T fast path: ov::element::dynamic means "not eligible" (decided once in
    // createPrimitive(), never re-checked per call). Only activates when QUERY/KEY_CACHE are
    // already natively bf16 (or natively f16) end-to-end -- no forced downcast of f32 data. The
    // AMX path reads exclusively from KEY_CACHE/VALUE_CACHE (never the transient current-step
    // KEY/VALUE inputs), so KEY_CACHE's negotiated precision -- not KEY's -- is what must match
    // QUERY and the target ISA.
    ov::element::Type m_amx_prec = ov::element::dynamic;
#ifdef OPENVINO_ARCH_X86_64
    // The JIT kernel requires its block_size (rows per AMX call) to be a multiple of 16, but real
    // key_block_size values (e.g. 4 for Qwen4) are usually smaller than that -- so AMX calls merge
    // m_amx_unit_blocks adjacent selected blocks (== 16 / gcd(16, key_block_size)) into one valid
    // 16-row call. m_gemv_qk1 therefore has block_size = m_amx_unit_blocks * key_block_size, and
    // m_gemv_qk4 has 4x that, to further amortize over longer runs of adjacent selected blocks --
    // never built at all if the physical page (KEY_CACHE's dim 2, static only) can't even fit one full unit.
    // Both kernels are safe to share read-only across worker threads: tile_config()/tile_release()
    // only touch per-thread AMX hardware state, and operator() only reads immutable JIT code plus
    // the pointers passed in. A run shorter than m_amx_unit_blocks (or the remainder after merging
    // as many full units as fit, or one capped at a physical-page boundary) falls back to the
    // scalar per-token path.
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk1;
    std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> m_gemv_qk4;
    size_t m_amx_unit_blocks = 0;
#endif
};

}  // namespace ov::intel_cpu::node