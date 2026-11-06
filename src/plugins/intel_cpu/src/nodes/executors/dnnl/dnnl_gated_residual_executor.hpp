// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/gated_residual.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}

namespace ov::intel_cpu {

/// @brief GatedResidual executor. The down/up/inject weight projections run through raw
/// `dnnl::matmul` primitives (with oneDNN native weight-decompression attributes for u8/i8/u4/i4
/// weights), the per-branch RMSNorm through the shared `jit_rms_kernel` (AVX2 minimum); everything else is scalar.
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
    void execute_read_or_final_mix(const MemoryArgs& memory);
    void execute_write(const MemoryArgs& memory);

    ExecutorContext::CPtr m_context;
    dnnl::stream m_stream;

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
    // Row-tile size (A+B+C = m_rbar+m_t+m_logit footprint bounded by a scratch budget, see
    // update()), so very long sequences never require O(rows) scratch memory. Every tile's GEMMs
    // always run the full m_tile_rows -- the last, possibly-partial tile pads with zeroed
    // scratch data and simply isn't stored back past its valid row count -- so each projection needs
    // exactly one dnnl::matmul primitive, regardless of how m_rows varies call to call.
    size_t m_tile_rows = 0;

    // Shared RMSNorm JIT kernel, compiled for one branch (data_size == hidden_size, per-element f32
    // gain) and invoked once per (row, branch). Always set for READ/FINAL_MIX (supports() requires AVX2).
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel;
    // WRITE mode: out = residual + gate * y for one (row, branch), JIT-compiled once per (precision, D).
    std::shared_ptr<kernel::JitKernelBase> m_write_kernel;
    // Mix over branches with the multiply-accumulate, 1/C scale and output conversion fused.
    std::shared_ptr<kernel::JitKernelBase> m_mix_kernel;
    // Same normalization with the GEMM-precision (bf16/f16) destination; only set for a bf16/f16 GEMM.
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel_gemm;

    // One projection's GEMM for a fixed M (rows), with the memory descriptors it was built for.
    struct ProjectionGemm {
        dnnl::matmul prim;
        dnnl::memory::desc src_md;
        dnnl::memory::desc weights_md;
        dnnl::memory::desc dst_md;
    };
    // The down/up/(inject) GEMMs for one M.
    struct TileGemms {
        size_t m = 0;
        ProjectionGemm down;
        ProjectionGemm up;
        ProjectionGemm inject;
    };
    // Dynamic M: the number of rows per inference varies, so GEMMs are created per bucketed M (see
    // normalize_rows() in the .cpp) through the runtime cache. Full tiles run `m_main` (M = tile size);
    // the last, partial tile runs `m_rem` (M = its bucketed row count) instead of padding up to a full tile.
    [[nodiscard]] TileGemms createTileGemms(size_t m, const MemoryArgs& memory);

    // Projection weights are graph constants (enforced by the node), so each projection is prepacked once into
    // the layout oneDNN prefers instead of running from the plain (K, N)-strided view (prefill ~29% and decode
    // ~28% faster on the production shape). The packed layout is identical for all M, so one copy per projection
    // serves every row count; the copy comes from the shared weights cache (as in GatherMatmul), so streams
    // share it. Compressed weights are not packed.
    enum ProjectionIdx : uint8_t { kDown = 0, kUp = 1, kInject = 2 };
    struct PackedWeights {
        bool enabled = false;
        bool ready = false;
        dnnl::memory::desc md;  // packed layout
        MemoryPtr data;         // owner of the packed bytes (shared through the weights cache)
        dnnl::memory view;      // oneDNN view over `data`
    };
    PackedWeights m_packed[3];
    TileGemms m_main;
    TileGemms m_rem;
    size_t m_rem_m = 0;

    // Per-tile scratch (f32 Rbar/Logit/T/WGLogit and, for a bf16/f16 GEMM, the downcast copies of Rbar and T)
    // lives in ONE block taken from the stream's shared scratchpad -- nodes execute sequentially, so a
    // single buffer serves every GatedResidual layer instead of each owning ~O(tile_rows*C*D) bytes. The
    // scratchpad is a single reusable block that other nodes also use (and may reallocate), so the typed
    // pointers below are re-derived from m_scratch at the start of every execute and the contents are
    // never assumed to survive between calls.
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

    // oneDNN views over the transposed scale/zero-point copies below. The scales/zero-points are graph constants
    // (enforced by the node), so they are built once, on the first execute (m_decomp_ready).
    dnnl::memory m_down_scale_mem;
    dnnl::memory m_down_zp_mem;
    dnnl::memory m_up_scale_mem;
    dnnl::memory m_up_zp_mem;
    dnnl::memory m_inject_scale_mem;
    dnnl::memory m_inject_zp_mem;
    bool m_decomp_ready = false;

    // f32 copies of the [G, N] decompression scale tensors (floating-point f16/bf16/f32 on the graph port) --
    // [G, N] row-major is the groups-outer physical order oneDNN's weights-decompression scale argument expects.
    std::vector<float> m_w_down_scale_f32;
    std::vector<float> m_w_up_scale_f32;
    std::vector<float> m_w_inject_scale_f32;

    // s32 copies of the [G, N] zero-point tensors (u4/i4 packed or u8/i8 on the graph port, the weights' type).
    std::vector<int32_t> m_w_down_zp;
    std::vector<int32_t> m_w_up_zp;
    std::vector<int32_t> m_w_inject_zp;
};

}  // namespace ov::intel_cpu
