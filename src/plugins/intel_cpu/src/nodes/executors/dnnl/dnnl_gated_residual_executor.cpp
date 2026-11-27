// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#    include <algorithm>
#    include <common/utils.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <functional>
#    include <memory>
#    include <utility>
#    include <vector>

#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_memory.h"
#    include "cpu_parallel.hpp"
#    include "cpu_shape.h"
#    include "dnnl_scratch_pad.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "nodes/common/blocked_desc_creator.h"
#    include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#    include "nodes/executors/dnnl/dnnl_gated_residual_executor.hpp"
#    include "nodes/executors/dnnl/dnnl_utils.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/fullyconnected_config.hpp"
#    include "nodes/executors/gated_residual_config.hpp"
#    include "nodes/executors/implementation_utils.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "nodes/kernels/gated_residual_mix.hpp"
#    include "nodes/kernels/x64/rms_kernel.hpp"
#    include "onednn/iml_type_mapper.h"
#    include "openvino/core/except.hpp"
#    include "openvino/core/shape.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/op/gated_residual.hpp"
#    include "openvino/runtime/system_conf.hpp"
#    include "post_ops.hpp"
#    include "utils/general_utils.h"

namespace ov::intel_cpu {

namespace {

using cpu_isa_t = dnnl::impl::cpu::x64::cpu_isa_t;

cpu_isa_t targetIsa() {
    return ov::with_cpu_x86_avx512_core() ? dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
}

// GEMM row count: M = 1 stays (oneDNN's GEMV path); M = 2..8 pads to 8, as from M >= 5 oneDNN prepacks compressed
// weights, which beats its in-kernel decompression of M = 2..4; M <= 16 pads to 16, larger M to a bucket.
size_t normalize_rows(size_t m) {
    constexpr size_t kDecodeRows = 8;
    constexpr size_t kSmallRows = 16;
    if (m == 1) {
        return m;
    }
    if (m <= kDecodeRows) {
        return kDecodeRows;
    }
    return m <= kSmallRows ? kSmallRows : utils::normalizeRowsBucket(m);
}

struct RmsKernelKey {
    ov::element::Type precision;  // source and gain element type
    ov::element::Type dst_precision;
    size_t data_size = 0;
    float eps = 0.0F;
    cpu_isa_t isa = dnnl::impl::cpu::x64::avx2;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;

        size_t seed = 0;
        seed = hash_combine(seed, precision.hash());
        seed = hash_combine(seed, dst_precision.hash());
        seed = hash_combine(seed, data_size);
        seed = hash_combine(seed, std::hash<float>{}(eps));
        seed = hash_combine(seed, static_cast<size_t>(isa));
        return seed;
    }

    bool operator==(const RmsKernelKey& rhs) const {
        return precision == rhs.precision && dst_precision == rhs.dst_precision && data_size == rhs.data_size &&
               eps == rhs.eps && isa == rhs.isa;
    }
};

}  // namespace

bool GatedResidualDnnlExecutor::supports(const GatedResidualConfig& config) {
    const auto it = config.descs.find(gatedResidualSrcArg(config.attrs.read_layout.residual));
    if (it == config.descs.end()) {
        return false;
    }
    if (none_of(it->second->getPrecision(), ov::element::f32, ov::element::bf16, ov::element::f16)) {
        return false;
    }
    // Every mode runs JIT kernels built for avx512_core and avx2 only.
    return ov::with_cpu_x86_avx2();
}

GatedResidualDnnlExecutor::GatedResidualDnnlExecutor(const GatedResidualAttrs& attrs,
                                                     const MemoryArgs& memory,
                                                     ExecutorContext::CPtr context)
    : m_context(std::move(context)),
      m_hc_count(attrs.hc_count),
      m_hidden_size(attrs.hidden_size),
      m_lowrank(attrs.lowrank),
      m_epsilon(attrs.epsilon),
      m_produce_write_gate(attrs.produce_write_gate),
      m_weights_compressed(attrs.read_layout.compressed),
      m_weights_have_zero_points(attrs.read_layout.has_zp),
      m_layout(attrs.read_layout) {
    m_precision = memory.at(gatedResidualSrcArg(attrs.read_layout.residual))->getPrecision();

    // Built at compile time: packs the constant weights.
    createProjections(attrs, memory);
}

size_t GatedResidualDnnlExecutor::maxTileRows() const {
    // Larger tiles amortize the weight re-read of every tile, up to a scratch budget. brgemm runs chunks of 128 rows x
    // 64 channels; the narrowest GEMM (down, L channels) limits parallelism, so a tile is as many 128-row blocks as
    // fit one wave of threads.
    constexpr size_t kBrgemmRowBlock = 128;
    constexpr size_t kBrgemmChannelBlock = 64;
    constexpr size_t kScratchBudgetBytes = 128UL << 20;
    const size_t C = m_hc_count;
    const size_t CD = C * m_hidden_size;
    const size_t bytes_per_row = (2 * CD) * sizeof(float) + m_lowrank * m_precision.size() +
                                 (m_produce_write_gate ? C * m_precision.size() : 0) +
                                 (m_precision != ov::element::f32 ? CD * sizeof(uint16_t) : 0);
    const size_t threads = static_cast<size_t>(std::max(1, CpuParallel::get_num_worker_threads()));
    const size_t channel_chunks = (m_lowrank + kBrgemmChannelBlock - 1) / kBrgemmChannelBlock;
    // The tile is a whole number of row blocks: as many as one wave of threads can take, within the scratch budget.
    const size_t blocks_per_wave = std::max<size_t>(1, threads / channel_chunks);
    const size_t blocks_in_budget = std::max<size_t>(1, kScratchBudgetBytes / (bytes_per_row * kBrgemmRowBlock));
    return std::min(blocks_per_wave, blocks_in_budget) * kBrgemmRowBlock;
}

void GatedResidualDnnlExecutor::createProjections(const GatedResidualAttrs& attrs, const MemoryArgs& memory) {
    using Type = ActivationPostOp::Type;
    const size_t C = m_hc_count;
    const size_t L = m_lowrank;
    const size_t CD = C * m_hidden_size;
    const float inv_c = 1.0F / static_cast<float>(C);
    const auto& engine = m_context->getEngine();
    const auto act = [](Type type, float alpha = 0.0F) {
        return ActivationPostOp(type, alpha, 0.0F, 0.0F);
    };

    // Scratchpads (see DnnlScratchPad): Rbar uses the executor's own block, the other tile buffers a child each, sized
    // once for a full tile. The projections' FC executors use the child the node put in their contexts.
    m_tile_rows = maxTileRows();
    const size_t tile = m_tile_rows;
    const bool gemm_lowp = m_precision != ov::element::f32;
    const auto pad = m_context->getScratchPad();
    const auto logit_pad = pad->child(kPadLogit);
    const auto t_pad = pad->child(kPadT);
    const auto wg_gate_pad = m_produce_write_gate ? pad->child(kPadWriteGate) : nullptr;
    const auto rbar_gemm_pad = gemm_lowp ? pad->child(kPadRbarGemm) : nullptr;
    const auto gemm_src_pad = gemm_lowp ? rbar_gemm_pad : pad;

    m_rbar = tileMemory(pad, ov::element::f32, tile, CD);
    m_logit = tileMemory(logit_pad, ov::element::f32, tile, CD);
    if (m_produce_write_gate) {
        m_wg_gate = tileMemory(wg_gate_pad, m_precision, tile, C);
    }
    if (gemm_lowp) {
        m_rbar_gemm = tileMemory(rbar_gemm_pad, m_precision, tile, CD);
    }

    struct Spec {
        size_t idx;
        size_t k;
        size_t n;
        size_t weights;
        size_t scales;
        size_t zero_points;
        DnnlScratchPadPtr src_pad;
        DnnlScratchPadPtr dst_pad;
        ov::element::Type dst_prc;
        PostOps post_ops;
    };
    std::vector<Spec> specs;
    // down: T = silu(down(Rbar) / C), stored at the GEMM precision for the up GEMM.
    specs.push_back({kDown,
                     CD,
                     L,
                     m_layout.w_down,
                     m_layout.scale_down,
                     m_layout.zp_down,
                     gemm_src_pad,
                     t_pad,
                     m_precision,
                     {act(Type::linear, inv_c), act(Type::swish, 1.0F)}});
    // up: gates = sigmoid(up(T)), in f32.
    specs.push_back({kUp,
                     L,
                     CD,
                     m_layout.w_up,
                     m_layout.scale_up,
                     m_layout.zp_up,
                     t_pad,
                     logit_pad,
                     ov::element::f32,
                     {act(Type::logistic)}});
    if (m_produce_write_gate) {
        // inject: write gate = 2 * sigmoid(inject(Rbar) / C), stored at the port precision.
        specs.push_back({kInject,
                         CD,
                         C,
                         m_layout.w_inject,
                         m_layout.scale_inject,
                         m_layout.zp_inject,
                         gemm_src_pad,
                         wg_gate_pad,
                         m_precision,
                         {act(Type::linear, inv_c), act(Type::logistic), act(Type::linear, 2.0F)}});
    }

    for (const auto& spec : specs) {
        auto& p = m_proj[spec.idx];
        OPENVINO_ASSERT(attrs.projection_contexts[spec.idx], "GatedResidual: missing projection executor context");
        p.context = attrs.projection_contexts[spec.idx];
        OPENVINO_ASSERT(p.context->getScratchPad() == pad->child(kPadProjections),
                        "GatedResidual: the projection executors must use the kPadProjections scratchpad");
        p.k = spec.k;
        p.n = spec.n;
        p.src_prc = m_precision;
        p.dst_prc = spec.dst_prc;
        p.src_pad = spec.src_pad;
        p.dst_pad = spec.dst_pad;
        p.attrs.postOps = spec.post_ops;
        p.attrs.dynamicQuantizationGroupSize = attrs.dynamic_quantization_group_size;
        p.attrs.modelType = attrs.model_type;

        p.args[ARG_WEI] = memory.at(gatedResidualSrcArg(spec.weights));
        p.args[ARG_BIAS] = std::make_shared<StaticMemory>(engine, MemoryDescUtils::makeEmptyDesc(), nullptr);
        if (m_weights_compressed) {
            p.args[ARG_WEI | ARG_ATTR_SCALES] = memory.at(gatedResidualSrcArg(spec.scales));
            if (m_weights_have_zero_points) {
                p.args[ARG_WEI | ARG_ATTR_ZERO_POINTS] = memory.at(gatedResidualSrcArg(spec.zero_points));
            }
        }
        // The executor of a full tile; the one of the remainder is bound in update().
        p.args[ARG_SRC] = tileMemory(p.src_pad, p.src_prc, tile, p.k);
        p.args[ARG_DST] = tileMemory(p.dst_pad, p.dst_prc, tile, p.n);
        p.full = CreateDnnlDefault<DnnlFCPrimitive, FCAttrs>{true, false}(p.attrs, p.args, p.context);
    }
}

MemoryPtr GatedResidualDnnlExecutor::tileMemory(const DnnlScratchPadPtr& pad,
                                                ov::element::Type prc,
                                                size_t rows,
                                                size_t cols) {
    const auto desc =
        BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp)->createSharedDesc(prc, Shape({rows, cols}));
    return pad->createScratchPadMem(desc);
}

void GatedResidualDnnlExecutor::bindFull(Projection& p) {
    if (!p.full_bound) {
        OPENVINO_ASSERT(p.full->update(p.args), "GatedResidual: projection executor update failed");
        p.full_bound = true;
    }
}

void GatedResidualDnnlExecutor::bindRemainder(Projection& p, size_t m) {
    if (p.part_m == m) {
        return;
    }
    p.part_args = p.args;
    p.part_args[ARG_SRC] = tileMemory(p.src_pad, p.src_prc, m, p.k);
    p.part_args[ARG_DST] = tileMemory(p.dst_pad, p.dst_prc, m, p.n);
    if (!p.part) {
        p.part = CreateDnnlDefault<DnnlFCPrimitive, FCAttrs>{true, false}(p.attrs, p.part_args, p.context);
    }
    OPENVINO_ASSERT(p.part->update(p.part_args), "GatedResidual: projection executor update failed");
    p.part_m = m;
}

impl_desc_type GatedResidualDnnlExecutor::implType() const {
    // The projections dominate: report the implementation chosen for the down GEMM.
    if (m_proj[kDown].full) {
        return m_proj[kDown].full->implType();
    }
    return targetIsa() == dnnl::impl::cpu::x64::avx512_core ? impl_desc_type::jit_avx512 : impl_desc_type::jit_avx2;
}

void GatedResidualDnnlExecutor::execute(const MemoryArgs& memory) {
    execute_read_or_final_mix(memory);
}

bool GatedResidualDnnlExecutor::update(const MemoryArgs& memory) {
    const cpu_isa_t isa = targetIsa();

    const size_t C = m_hc_count;
    const size_t D = m_hidden_size;
    const size_t CD = C * D;

    const auto& residual_dims = memory.at(gatedResidualSrcArg(m_layout.residual))->getStaticDims();
    m_rows = shape_size(residual_dims) / CD;
    if (m_rows == 0) {
        return true;
    }

    if (!m_rms_kernel) {
        // The kernels depend only on (precision, dst precision, D, eps, ISA), so layers with the same hidden size share
        // them through the runtime cache.
        const auto get_kernel = [&](ov::element::Type dst_prc) {
            const RmsKernelKey key{m_precision, dst_prc, D, m_epsilon, isa};
            return m_context->getRuntimeCache()
                ->getOrCreate(
                    key,
                    [](const RmsKernelKey& k) -> std::shared_ptr<kernel::JitKernelBase> {
                        kernel::jit_rms_compile_params jcp;
                        jcp.src_prc = k.precision;
                        jcp.dst_prc = k.dst_precision;
                        jcp.data_size = k.data_size;
                        jcp.scale_size = k.data_size;
                        jcp.eps = k.eps;
                        // The gain is read in the port precision.
                        jcp.scale_prc = k.precision;
                        std::shared_ptr<kernel::JitKernelBase> kernel;
                        if (k.isa == dnnl::impl::cpu::x64::avx512_core) {
                            kernel = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx512_core>>(jcp);
                        } else {
                            kernel = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx2>>(jcp);
                        }
                        kernel->create_kernel();
                        return kernel;
                    })
                .first;
        };
        // The mix reads the f32 Rbar; a GEMM precision other than f32 needs a second copy of the normalization.
        m_rms_kernel = get_kernel(ov::element::f32);
        if (m_precision != ov::element::f32) {
            m_rms_kernel_gemm = get_kernel(m_precision);
        }
    }

    const size_t part_rows = m_rows % m_tile_rows;
    m_part_m = part_rows != 0 ? std::min(normalize_rows(part_rows), m_tile_rows) : 0;
    const bool part_is_full = m_part_m == m_tile_rows;
    for (auto& p : m_proj) {
        if (!p.full) {
            continue;
        }
        if (m_rows >= m_tile_rows || part_is_full) {
            bindFull(p);
        }
        if (m_part_m != 0 && !part_is_full) {
            bindRemainder(p, m_part_m);
        }
    }
    return true;
}

void GatedResidualDnnlExecutor::execute_read_or_final_mix(const MemoryArgs& memory) {
    const size_t C = m_hc_count;
    const size_t D = m_hidden_size;
    const size_t CD = C * D;
    const size_t rows = m_rows;
    if (rows == 0) {
        return;
    }

    // The shared scratchpad block may have been reallocated since the last call: read the data pointers again.
    auto* rbar = m_rbar->getDataAs<float>();
    const auto* logit = m_logit->getDataAs<const float>();
    const bool gemm_lowp = m_precision != ov::element::f32;
    auto* rbar_gemm = gemm_lowp ? m_rbar_gemm->getDataAs<uint16_t>() : nullptr;

    // Inputs and outputs may have leading batch dims; rows are indexed flat.
    auto* mixed_out = memory.at(gatedResidualDstArg(0))->getDataAs<uint8_t>();  // [rows, D] at the port precision

    // The RMS kernel reads the residual and the gain in their port precision.
    const auto* residual_bytes = memory.at(gatedResidualSrcArg(m_layout.residual))->getDataAs<const uint8_t>();
    const size_t residual_elem_size = m_precision.size();
    const auto* gain_bytes = memory.at(gatedResidualSrcArg(m_layout.norm_weight))->getDataAs<const uint8_t>();

    const bool produce_write_gate = m_produce_write_gate;
    uint8_t* write_gate_out = nullptr;  // [rows, C] output port, contiguous ncsp at the port precision
    if (produce_write_gate) {
        write_gate_out = memory.at(gatedResidualDstArg(1))->getDataAs<uint8_t>();
    }

    const size_t tile_rows = m_tile_rows;

    // Full tiles run tile_rows GEMM rows, the last m_part_m. `cur_rows` bounds the elementwise steps and stores;
    // `gemm_rows` may be larger (bucketed), with the extra source rows zeroed.
    const bool part_is_full = m_part_m == tile_rows;
    for (size_t tile_start = 0; tile_start < rows; tile_start += tile_rows) {
        const size_t cur_rows = std::min(tile_rows, rows - tile_start);
        const bool full_tile = cur_rows == tile_rows;
        const size_t gemm_rows = full_tile ? tile_rows : m_part_m;
        const bool on_full = full_tile || part_is_full;
        const auto run = [&](Projection& p) {
            if (on_full) {
                p.full->execute(p.args);
            } else {
                p.part->execute(p.part_args);
            }
        };

        // Step 1: per-row RMSNorm per branch: the f32 Rbar for the mix and a GEMM-precision copy for the GEMMs.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            float* rbar_row = rbar + row * CD;
            const auto* src_row = residual_bytes + (tile_start + row) * CD * residual_elem_size;
            for (size_t c = 0; c < C; c++) {
                kernel::jit_rms_call_args args{src_row + c * D * residual_elem_size,
                                               gain_bytes + c * D * residual_elem_size,
                                               reinterpret_cast<uint8_t*>(rbar_row + c * D)};
                (*m_rms_kernel)(&args);
                if (gemm_lowp) {
                    args.dst = reinterpret_cast<uint8_t*>(rbar_gemm + row * CD + c * D);
                    (*m_rms_kernel_gemm)(&args);
                }
            }
        });

        // Zero the GEMM padding rows (the shared scratchpad holds other nodes' data); their results are never stored.
        if (cur_rows < gemm_rows) {
            std::fill(rbar + cur_rows * CD, rbar + gemm_rows * CD, 0.0F);
            if (gemm_lowp) {
                std::fill(rbar_gemm + cur_rows * CD, rbar_gemm + gemm_rows * CD, uint16_t{0});
            }
        }

        // Step 2: T[M,L] = silu(Rbar[M,CD] @ w_down[L,CD]^T / C).
        run(m_proj[kDown]);

        // Step 3: Gate[M,CD] = sigmoid(T[M,L] @ w_up[CD,L]^T).
        run(m_proj[kUp]);

        // Step 4: out = (1/C) * sum_c gate_c * rbar_c, one kernel call per row, converted to the output precision.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            ov::Extensions::Cpu::XARCH::gated_residual_mix(logit + row * CD,
                                                           rbar + row * CD,
                                                           mixed_out + (tile_start + row) * D * residual_elem_size,
                                                           m_precision,
                                                           D,
                                                           C,
                                                           1.0F / static_cast<float>(C));
        });

        // Step 5: WriteGate[M,C] = 2 * sigmoid(Rbar @ w_inject^T / C). The GEMM writes a fixed scratch buffer; only the
        // valid rows are copied out.
        if (produce_write_gate) {
            run(m_proj[kInject]);
            const size_t wg_elem_size = m_precision.size();
            std::memcpy(write_gate_out + tile_start * C * wg_elem_size,
                        m_wg_gate->getData(),
                        cur_rows * C * wg_elem_size);
        }
    }
}

}  // namespace ov::intel_cpu

#endif  // OPENVINO_ARCH_X86_64
