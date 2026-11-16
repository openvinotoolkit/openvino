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

/// GatedResidual executor (x86-64). The projections run through FullyConnected executors (oneDNN weight decompression);
/// RMSNorm, mix and write run through JIT kernels (AVX2 minimum). Accumulation is f32.
class GatedResidualDnnlExecutor : public Executor {
public:
    static bool supports(const GatedResidualConfig& config);

    GatedResidualDnnlExecutor(const GatedResidualAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    void createProjections(const GatedResidualAttrs& attrs, const MemoryArgs& memory);
    [[nodiscard]] size_t maxTileRows() const;
    // [rows, cols] tensor at `offset` bytes into the shared scratchpad block.
    [[nodiscard]] MemoryPtr tileMemory(size_t offset, ov::element::Type prc, size_t rows, size_t cols) const;
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

    // Element type of the ports.
    ov::element::Type m_precision = ov::element::f32;
    // Element type of the projection GEMM inputs and of T: m_precision, except f32 for compressed weights where oneDNN
    // cannot decompress at that precision (as in the FC node).
    ov::element::Type m_gemm_precision = ov::element::f32;

    bool m_weights_compressed = false;
    bool m_weights_have_zero_points = false;
    ov::op::internal::GatedResidual::ReadLayout m_layout{};

    size_t m_rows = 0;
    // Rows per full tile. The last tile runs a GEMM padded to a bucketed row count (m_part_m); padding rows are zeroed
    // and never stored.
    size_t m_tile_rows = 0;
    size_t m_part_m = 0;

    // RMSNorm kernel for one branch, run per (row, branch).
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel;
    // WRITE: out = residual + gate * y for one (row, branch).
    std::shared_ptr<kernel::JitKernelBase> m_write_kernel;
    // Mix over branches: out = (1/C) * sum_c gate_c * rbar_c, converted to the output precision.
    std::shared_ptr<kernel::JitKernelBase> m_mix_kernel;
    // The same normalization stored as the GEMM input precision; only set when that differs from f32.
    std::shared_ptr<kernel::JitKernelBase> m_rms_kernel_gemm;

    enum ProjectionIdx : uint8_t { kDown = 0, kUp = 1, kInject = 2 };

    // FC executor of one projection for one row count, bound to the tile buffers in update().
    struct TileExecutor {
        ExecutorPtr executor;
        MemoryArgs args;
        size_t m = 0;  // rows it is bound to, 0 until then
    };

    // One projection. oneDNN picks the packed weights layout; the weights are packed once at construction through the
    // context's private cache.
    struct Projection {
        ExecutorContext::CPtr context;  // private weight cache, scratchpad behind the tile buffers
        FCAttrs attrs;
        MemoryArgs args;  // weights, bias and decompression parameters
        size_t k = 0;
        size_t n = 0;
        ov::element::Type src_prc;
        ov::element::Type dst_prc;
        size_t src_offset = 0;  // tile buffers, in bytes from the start of the scratchpad block
        size_t dst_offset = 0;
        TileExecutor full;     // full tiles
        TileExecutor partial;  // the last tile, m_part_m rows
    };
    Projection m_proj[3];
    void bindProjection(Projection& p, TileExecutor& tile, size_t m);

    // Per-tile buffers: views of the shared scratchpad block, sized once for a full tile. Other nodes may reallocate
    // the block and contents do not survive calls, so read data pointers on every execute.
    MemoryPtr m_rbar;       // f32 [tile, C*D], read by the mix
    MemoryPtr m_logit;      // f32 [tile, C*D], the up projection output
    MemoryPtr m_wg_gate;    // port precision [tile, C], the inject projection output
    MemoryPtr m_rbar_gemm;  // GEMM precision [tile, C*D], only when that is not f32
};

}  // namespace ov::intel_cpu
