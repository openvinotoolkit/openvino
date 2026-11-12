// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}

namespace ov::intel_cpu {

/// @brief GatedResidual executor. The down/up/inject weight projections run through the plugin's FullyConnected
/// executors (`DnnlFCPrimitive`, with oneDNN native weight decompression for u8/i8/u4/i4 weights), the per-branch
/// RMSNorm through the shared `jit_rms_kernel` (AVX2 minimum); everything else is scalar.
/// Accumulation is always f32 regardless of the port precision (f32/bf16/f16).
class GatedResidualDnnlExecutor : public Executor {
public:
    static bool supports(const GatedResidualConfig& config);

    GatedResidualDnnlExecutor(const GatedResidualAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    void createProjections(const GatedResidualAttrs& attrs, const MemoryArgs& memory);
    [[nodiscard]] size_t maxTileRows() const;
    [[nodiscard]] size_t fcScratchBytes(size_t m) const;
    void execute_read_or_final_mix(const MemoryArgs& memory);
    void execute_write(const MemoryArgs& memory);

    ExecutorContext::CPtr m_context;

    ov::op::internal::GatedResidual::Mode m_mode = ov::op::internal::GatedResidual::Mode::READ;
    size_t m_hc_count = 0;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    float m_epsilon = 1e-6F;
    bool m_norm_gain_plus_one = true;
    bool m_produce_write_gate = false;

    ov::element::Type m_precision = ov::element::f32;
    ov::element::Type m_gemm_precision = ov::element::f32;

    bool m_weights_compressed = false;
    bool m_weights_have_zero_points = false;
    ov::op::internal::GatedResidual::ReadLayout m_layout{};

    size_t m_rows = 0;
    // Rows per full tile (see maxTileRows()). A partial last tile runs a GEMM padded to a bucketed row count with
    // zeroed scratch rows, which are never stored back.
    size_t m_tile_rows = 0;

    // Shared RMSNorm JIT kernel, compiled for one branch (data_size == hidden_size) and invoked once per (row, branch).
    // Always set for READ/FINAL_MIX (supports() requires AVX2).
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel;
    // WRITE mode: out = residual + gate * y for one (row, branch), JIT-compiled once per (precision, D).
    std::shared_ptr<kernel::JitKernelBase> m_write_kernel;
    // Mix over branches with the multiply-accumulate, 1/C scale and output conversion fused.
    std::shared_ptr<kernel::JitKernelBase> m_mix_kernel;
    // Same normalization with the GEMM-precision (bf16/f16) destination; only set for a bf16/f16 GEMM.
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel_gemm;

    enum ProjectionIdx : uint8_t { kDown = 0, kUp = 1, kInject = 2 };

    // One projection (down / up / inject) run by the plugin's FullyConnected executor: oneDNN picks the weights
    // layout, the weights are repacked into it once (at construction, through the shared weights cache), compressed
    // weights are decompressed by oneDNN and the epilogue is expressed as FC post-ops.
    struct Projection {
        // The projection's own context (see GatedResidualAttrs::projection_contexts).
        ExecutorContext::CPtr context;
        ExecutorPtr executor;
        FCAttrs attrs;
        DnnlShapeAgnosticDataPtr shapeAgnosticData;  // the same attributes the executor was built with
        MemoryArgs args;  // weights, bias and decompression parameters, plus src/dst of the last update
        size_t k = 0;
        size_t n = 0;
        ov::element::Type src_prc;
        ov::element::Type dst_prc;
        // Arguments of the last executor update; it is repeated only when one of them changes.
        size_t m = 0;
        const void* src = nullptr;
        const void* dst = nullptr;
    };
    Projection m_proj[3];
    // Binds the projection to src/dst for m rows (when they changed) and runs it.
    void runProjection(Projection& p, size_t m, void* src, void* dst);

    // Row counts of the GEMMs, bucketed by normalize_rows() in the .cpp: full tiles run m_main_m rows, the last
    // partial tile m_rem_m (0: none).
    size_t m_main_m = 0;
    size_t m_rem_m = 0;
    // The FC primitives take their scratchpad from the same shared block as m_scratch, starting at its beginning: the
    // node's buffers start after this many bytes.
    size_t m_fc_scratch_bytes = 0;

    // Per-tile scratch (f32 Rbar and Logit, T, the write gate and, for a bf16/f16 GEMM, a copy of Rbar) is one block of
    // the stream's shared scratchpad: nodes execute sequentially, so one buffer serves every GatedResidual layer.
    // Other nodes share and may reallocate that block, so the pointers below are re-derived from m_scratch on every
    // execute and the contents never survive between calls.
    MemoryPtr m_scratch;
    size_t m_off_rbar = 0;
    size_t m_off_logit = 0;
    size_t m_off_t = 0;
    size_t m_off_wg_gate = 0;
    size_t m_off_rbar_gemm = 0;
    float* m_rbar = nullptr;
    float* m_logit = nullptr;
    uint8_t* m_t = nullptr;        // T = silu(down(Rbar)/C) at the GEMM precision
    uint8_t* m_wg_gate = nullptr;  // write gate at the port precision, one row per tile row
    uint16_t* m_rbar_gemm = nullptr;
};

}  // namespace ov::intel_cpu
