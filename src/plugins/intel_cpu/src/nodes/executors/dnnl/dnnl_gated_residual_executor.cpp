// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#    include <algorithm>
#    include <common/primitive_hashing.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <functional>
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_common.hpp>
#    include <vector>

#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_memory.h"
#    include "cpu_shape.h"
#    include "memory_desc/cpu_blocked_memory_desc.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "memory_desc/cpu_memory_desc_utils.h"
#    include "nodes/common/blocked_desc_creator.h"
#    include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#    include "nodes/executors/dnnl/dnnl_gated_residual_executor.hpp"
#    include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/fullyconnected_config.hpp"
#    include "nodes/executors/gated_residual_config.hpp"
#    include "nodes/executors/implementation_utils.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "nodes/kernels/x64/gated_residual_mix_kernel.hpp"
#    include "nodes/kernels/x64/gated_residual_write_kernel.hpp"
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

using ReadInputIdx = ov::op::internal::GatedResidual::ReadInputIdx;
using WriteInputIdx = ov::op::internal::GatedResidual::WriteInputIdx;
using Mode = ov::op::internal::GatedResidual::Mode;

// Brings a decompression parameter port (scales or zero-points: [G, N] with N contiguous, any of f16/bf16/f32/u8/i8/
// u4/i4) into the logical [OC, G] row-major f32 tensor the FC path expects. One oneDNN reorder does both the transpose
// and the conversion: the source is described as the logical [N, G] tensor whose physical layout is [G, N].
MemoryPtr make_decompression_param(const MemoryPtr& port, const dnnl::engine& engine) {
    const auto& dims = port->getStaticDims();
    OPENVINO_ASSERT(dims.size() == 2, "GatedResidual: decompression parameters must be rank 2, got ", dims.size());
    const size_t groups = dims[0];
    const size_t n = dims[1];
    const CpuBlockedMemoryDesc src_desc(port->getPrecision(), Shape(VectorDims{n, groups}), {groups, n}, {1, 0});
    const Memory src(engine, src_desc, port->getData());
    auto dst = std::make_shared<Memory>(engine, CpuBlockedMemoryDesc(ov::element::f32, Shape(VectorDims{n, groups})));
    dst->load(src, false, false);
    return dst;
}

// Largest row tile.
constexpr size_t kMaxTileRows = 512;

// Rounds a row count up to a bucket, bounding the number of distinct GEMM shapes (and so cached primitives
// and JIT builds) a dynamic row count can produce.
size_t normalize_rows(size_t m) {
    const auto rnd_up = [](size_t v, size_t multiple) {
        return (v + multiple - 1) / multiple * multiple;
    };
    // Tiny M (decode): powers of two from 2, so a handful of real rows does not run the GEMM epilogues (SiLU,
    // sigmoid, write gate) over a 16-row tile. M=1 is excluded on purpose: oneDNN picks a GEMV-specific packed weights
    // layout for it, while every M >= 2 shares one layout, so M=1 would need a second packed copy of the weights.
    if (m <= 16) {
        size_t bucket = 2;
        while (bucket < m) {
            bucket <<= 1;
        }
        return bucket;
    }
    if (m < 512) {
        return rnd_up(m, 16);
    }
    if (m < 1024) {
        return rnd_up(m, 32);
    }
    return rnd_up(m, 256);
}

struct WriteKernelKey {
    ov::element::Type precision;
    size_t data_size = 0;
    bool avx512 = false;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, precision.hash());
        seed = hash_combine(seed, data_size);
        seed = hash_combine(seed, static_cast<size_t>(avx512));
        return seed;
    }

    bool operator==(const WriteKernelKey& rhs) const {
        return precision == rhs.precision && data_size == rhs.data_size && avx512 == rhs.avx512;
    }
};

struct MixKernelKey {
    ov::element::Type precision;
    size_t data_size = 0;
    size_t branches = 0;
    bool avx512 = false;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, precision.hash());
        seed = hash_combine(seed, data_size);
        seed = hash_combine(seed, branches);
        seed = hash_combine(seed, static_cast<size_t>(avx512));
        return seed;
    }

    bool operator==(const MixKernelKey& rhs) const {
        return precision == rhs.precision && data_size == rhs.data_size && branches == rhs.branches &&
               avx512 == rhs.avx512;
    }
};

struct RmsKernelKey {
    ov::element::Type precision;  // source and gain element type
    ov::element::Type dst_precision;
    size_t data_size = 0;
    float eps = 0.0F;
    bool plus_one = false;  // gain is applied as (1 + weight)
    bool avx512 = false;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, precision.hash());
        seed = hash_combine(seed, dst_precision.hash());
        seed = hash_combine(seed, data_size);
        seed = hash_combine(seed, std::hash<float>{}(eps));
        seed = hash_combine(seed, static_cast<size_t>(plus_one));
        seed = hash_combine(seed, static_cast<size_t>(avx512));
        return seed;
    }

    bool operator==(const RmsKernelKey& rhs) const {
        return precision == rhs.precision && dst_precision == rhs.dst_precision && data_size == rhs.data_size &&
               eps == rhs.eps && plus_one == rhs.plus_one && avx512 == rhs.avx512;
    }
};

}  // namespace

bool GatedResidualDnnlExecutor::supports(const GatedResidualConfig& config) {
    const auto it = config.descs.find(gatedResidualSrcArg(ov::op::internal::GatedResidual::ReadInputIdx::RESIDUAL));
    if (it == config.descs.end()) {
        return false;
    }
    if (none_of(it->second->getPrecision(), ov::element::f32, ov::element::bf16, ov::element::f16)) {
        return false;
    }
    using dnnl::impl::cpu::x64::cpu_isa_t;
    using dnnl::impl::cpu::x64::mayiuse;
    // Every mode runs JIT kernels (RMSNorm / mix / write) that are instantiated for avx512_core and avx2 only.
    if (!mayiuse(cpu_isa_t::avx2)) {
        return false;
    }
    // Uncompressed weights are fed to the GEMM at the port precision (no upconvert path), so a bf16/f16 port needs
    // native bf16/f16 support. Compressed weights are decompressed by oneDNN and may still run the GEMM in f32.
    const auto prc = it->second->getPrecision();
    if (config.attrs.mode != ov::op::internal::GatedResidual::Mode::WRITE && !config.attrs.read_layout.compressed) {
        if ((prc == ov::element::bf16 && !mayiuse(cpu_isa_t::avx512_core_bf16)) ||
            (prc == ov::element::f16 && !mayiuse(cpu_isa_t::avx512_core_fp16))) {
            return false;
        }
    }
    return true;
}

GatedResidualDnnlExecutor::GatedResidualDnnlExecutor(const GatedResidualAttrs& attrs,
                                                     const MemoryArgs& memory,
                                                     const ExecutorContext::CPtr& context)
    : m_context(context),
      m_mode(attrs.mode),
      m_hc_count(attrs.hc_count),
      m_hidden_size(attrs.hidden_size),
      m_lowrank(attrs.lowrank),
      m_epsilon(attrs.epsilon),
      m_norm_gain_plus_one(attrs.norm_gain_plus_one),
      m_produce_write_gate(attrs.produce_write_gate),
      m_weights_compressed(attrs.read_layout.compressed),
      m_weights_have_zero_points(attrs.read_layout.has_zp),
      m_layout(attrs.read_layout) {
    m_precision = memory.at(gatedResidualSrcArg(ReadInputIdx::RESIDUAL))->getPrecision();

    // The GEMM runs at bf16/f16 (src and weights; accumulation stays f32) only when the port precision is bf16/f16
    // and the CPU ISA accelerates it at that precision; otherwise it runs at f32.
    using dnnl::impl::cpu::x64::cpu_isa_t;
    using dnnl::impl::cpu::x64::mayiuse;
    if (m_precision == ov::element::bf16 && mayiuse(cpu_isa_t::avx512_core_bf16)) {
        m_gemm_precision = ov::element::bf16;
    } else if (m_precision == ov::element::f16 && mayiuse(cpu_isa_t::avx512_core_fp16)) {
        m_gemm_precision = ov::element::f16;
    }
    // The executors are built here, at compile time, and repack the constant weights into the layout oneDNN picks.
    createProjections(attrs, memory);
}

size_t GatedResidualDnnlExecutor::maxTileRows() const {
    // Rows run in tiles; each tile re-streams the weights, so larger tiles amortize that traffic, up to a scratch
    // budget in the shared scratchpad. The brgemm inner product blocks rows by 128 only when their count is a
    // multiple of 128, so tiles are multiples of 128; 512 rows measured fastest among 256..768.
    constexpr size_t kMinTileRows = 32;
    constexpr size_t kBrgemmRowBlock = 128;
    constexpr size_t kScratchBudgetBytes = 128UL << 20;
    const size_t C = m_hc_count;
    const size_t CD = C * m_hidden_size;
    // Per-row footprint of every buffer reserved in update(): rbar and logit (f32), t (GEMM precision), the optional
    // write-gate output (data precision) and the optional low-precision copy of rbar for the GEMM.
    const size_t bytes_per_row = (2 * CD) * sizeof(float) + m_lowrank * m_gemm_precision.size() +
                                 (m_produce_write_gate ? C * m_precision.size() : 0) +
                                 (m_gemm_precision != ov::element::f32 ? CD * sizeof(uint16_t) : 0);
    const size_t tile = std::clamp(kScratchBudgetBytes / bytes_per_row, kMinTileRows, kMaxTileRows);
    // A small budget falls back to a 32-row granularity (every normalized remainder stays <= the tile).
    const size_t alignment = tile >= kBrgemmRowBlock ? kBrgemmRowBlock : kMinTileRows;
    return tile - tile % alignment;
}

void GatedResidualDnnlExecutor::createProjections(const GatedResidualAttrs& attrs, const MemoryArgs& memory) {
    if (m_mode == Mode::WRITE) {
        return;
    }
    using Type = ActivationPostOp::Type;
    const size_t C = m_hc_count;
    const size_t L = m_lowrank;
    const size_t CD = C * m_hidden_size;
    const float inv_c = 1.0F / static_cast<float>(C);
    const auto& engine = m_context->getEngine();
    const auto act = [](Type type, float alpha = 0.0F) {
        return ActivationPostOp(type, alpha, 0.0F, 0.0F);
    };

    struct Spec {
        size_t idx;
        size_t k;
        size_t n;
        size_t weights;
        size_t scales;
        size_t zero_points;
        ov::element::Type dst_prc;
        PostOps post_ops;
    };
    std::vector<Spec> specs;
    // The down GEMM produces T = silu(down(Rbar) / C), stored at the GEMM precision so the up GEMM reads it as-is.
    specs.push_back({kDown,
                     CD,
                     L,
                     ReadInputIdx::W_DOWN,
                     ReadInputIdx::SCALE_DOWN,
                     ReadInputIdx::ZP_DOWN,
                     m_gemm_precision,
                     {act(Type::linear, inv_c), act(Type::swish, 1.0F)}});
    // The up GEMM produces the gates directly: logit -> sigmoid in the epilogue.
    specs.push_back({kUp,
                     L,
                     CD,
                     ReadInputIdx::W_UP,
                     ReadInputIdx::SCALE_UP,
                     ReadInputIdx::ZP_UP,
                     ov::element::f32,
                     {act(Type::logistic)}});
    if (m_produce_write_gate) {
        // The write gate 2 * sigmoid(logit / C) is the GEMM epilogue, stored at the output port's precision so
        // the valid rows can be copied to the output as-is.
        specs.push_back({kInject,
                         CD,
                         C,
                         m_layout.w_inject,
                         m_layout.scale_inject,
                         m_layout.zp_inject,
                         m_precision,
                         {act(Type::linear, inv_c), act(Type::logistic), act(Type::linear, 2.0F)}});
    }

    const auto& creator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
    // src/dst only give the executors the row count of the main tile; the buffers are never touched and are dropped
    // right after.
    const size_t main_m = normalize_rows(maxTileRows());
    for (const auto& spec : specs) {
        auto& p = m_proj[spec.idx];
        p.context = attrs.projection_contexts[spec.idx];
        OPENVINO_ASSERT(p.context, "GatedResidual: missing projection executor context");
        p.k = spec.k;
        p.n = spec.n;
        p.src_prc = m_gemm_precision;
        p.dst_prc = spec.dst_prc;
        p.attrs.postOps = spec.post_ops;

        p.args[ARG_WEI] = memory.at(gatedResidualSrcArg(spec.weights));
        p.args[ARG_BIAS] = std::make_shared<StaticMemory>(engine, MemoryDescUtils::makeEmptyDesc(), nullptr);
        if (m_weights_compressed) {
            p.args[ARG_WEI | ARG_ATTR_SCALES] =
                make_decompression_param(memory.at(gatedResidualSrcArg(spec.scales)), engine);
            if (m_weights_have_zero_points) {
                p.args[ARG_WEI | ARG_ATTR_ZERO_POINTS] =
                    make_decompression_param(memory.at(gatedResidualSrcArg(spec.zero_points)), engine);
            }
        }
        p.args[ARG_SRC] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.src_prc, Shape({main_m, p.k})));
        p.args[ARG_DST] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.dst_prc, Shape({main_m, p.n})));

        p.shapeAgnosticData = DnnlFCPrimitive::createShapeAgnosticData(p.attrs, p.args, p.context, false);
        p.executor = CreateDnnlDefault<DnnlFCPrimitive, FCAttrs>{true, false}(p.attrs, p.args, p.context);
        p.args.erase(ARG_SRC);
        p.args.erase(ARG_DST);
    }
}

size_t GatedResidualDnnlExecutor::fcScratchBytes(size_t m) const {
    const auto& creator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
    const auto& engine = m_context->getEngine();
    size_t bytes = 0;
    for (const auto& p : m_proj) {
        if (!p.executor) {
            continue;
        }
        // Same primitive the executor creates on its update (a runtime cache hit then).
        MemoryArgs args = p.args;
        args[ARG_SRC] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.src_prc, Shape({m, p.k})));
        args[ARG_DST] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.dst_prc, Shape({m, p.n})));
        const auto primitive = DnnlFCPrimitive::create(args, p.attrs, p.context, p.shapeAgnosticData);
        bytes = std::max(bytes, primitive->scratchPadDesc()->getCurrentMemSize());
    }
    return (bytes + 63) / 64 * 64;
}

void GatedResidualDnnlExecutor::runProjection(Projection& p, size_t m, void* src, void* dst) {
    if (p.m != m || p.src != src || p.dst != dst) {
        // The executor binds the buffers when it is updated. They change with the row count, or when the shared
        // scratchpad block has been reallocated.
        const auto& creator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        const auto& engine = m_context->getEngine();
        p.args[ARG_SRC] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.src_prc, Shape({m, p.k})), src);
        p.args[ARG_DST] = std::make_shared<Memory>(engine, creator->createSharedDesc(p.dst_prc, Shape({m, p.n})), dst);
        OPENVINO_ASSERT(p.executor->update(p.args), "GatedResidual: projection executor update failed");
        p.m = m;
        p.src = src;
        p.dst = dst;
    }
    p.executor->execute(p.args);
}

impl_desc_type GatedResidualDnnlExecutor::implType() const {
    // READ / FINAL_MIX are dominated by the projections: report what oneDNN chose for the down GEMM (the largest one).
    if (m_proj[kDown].executor) {
        return m_proj[kDown].executor->implType();
    }
    // WRITE has no GEMM, only the JIT elementwise kernel.
    return ov::with_cpu_x86_avx512_core() ? impl_desc_type::jit_avx512 : impl_desc_type::jit_avx2;
}

void GatedResidualDnnlExecutor::execute(const MemoryArgs& memory) {
    if (m_mode == Mode::WRITE) {
        execute_write(memory);
    } else {
        execute_read_or_final_mix(memory);
    }
}

bool GatedResidualDnnlExecutor::update(const MemoryArgs& memory) {
    if (m_mode == Mode::WRITE) {
        // execute_write() indexes Y and the gate once per residual row, so they must have the same number of rows
        // (the op validates this for static shapes; dynamic dimensions are only known here).
        const auto rows = shape_size(memory.at(gatedResidualSrcArg(WriteInputIdx::RESIDUAL))->getStaticDims()) /
                          (m_hc_count * m_hidden_size);
        OPENVINO_ASSERT(
            shape_size(memory.at(gatedResidualSrcArg(WriteInputIdx::SUBLAYER_Y))->getStaticDims()) ==
                    rows * m_hidden_size &&
                shape_size(memory.at(gatedResidualSrcArg(WriteInputIdx::WRITE_GATE))->getStaticDims()) ==
                    rows * m_hc_count,
            "GatedResidual (WRITE): sub-layer output and write gate must have the same leading dimensions as the "
            "residual");
        if (!m_write_kernel) {
            // Depends only on (precision, D, ISA): shared by every WRITE layer through the runtime cache.
            const WriteKernelKey key{m_precision, m_hidden_size, ov::with_cpu_x86_avx512_core()};
            m_write_kernel =
                m_context->getRuntimeCache()
                    ->getOrCreate(
                        key,
                        [](const WriteKernelKey& k) -> std::shared_ptr<kernel::JitKernelBase> {
                            kernel::jit_gated_residual_write_compile_params jcp;
                            jcp.prc = k.precision;
                            jcp.data_size = k.data_size;
                            // supports() guarantees AVX2.
                            std::shared_ptr<kernel::JitKernelBase> kernel;
                            if (k.avx512) {
                                kernel = std::make_shared<
                                    kernel::jit_gated_residual_write_kernel<dnnl::impl::cpu::x64::avx512_core>>(jcp);
                            } else {
                                kernel = std::make_shared<
                                    kernel::jit_gated_residual_write_kernel<dnnl::impl::cpu::x64::avx2>>(jcp);
                            }
                            kernel->create_kernel();
                            return kernel;
                        })
                    .first;
        }
        return true;
    }

    const size_t C = m_hc_count;
    const size_t D = m_hidden_size;
    const size_t L = m_lowrank;
    const size_t CD = C * D;

    const auto& residual_dims = memory.at(gatedResidualSrcArg(ReadInputIdx::RESIDUAL))->getStaticDims();
    m_rows = shape_size(residual_dims) / CD;
    if (m_rows == 0) {
        return true;
    }

    if (!m_rms_kernel) {
        // The kernels only depend on (precision, dst precision, D, eps, ISA), so every GatedResidual layer with the
        // same hidden size shares them through the stream's runtime cache. For a bf16/f16 GEMM two instances run per
        // branch: one writes the f32 Rbar (the mix reads it), the other the GEMM's bf16/f16 source directly, from the
        // same f32 computation.
        const bool avx512 = ov::with_cpu_x86_avx512_core();
        const auto get_kernel = [&](ov::element::Type dst_prc) {
            const RmsKernelKey key{m_precision, dst_prc, D, m_epsilon, m_norm_gain_plus_one, avx512};
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
                        // The gain is read at its port precision (same as the residual) and, for
                        // norm_gain_plus_one, biased by +1 inside the kernel: no f32 copy of the gain.
                        jcp.scale_prc = k.precision;
                        jcp.scale_plus_one = k.plus_one;
                        // supports() guarantees AVX2 for every mode that reaches here.
                        std::shared_ptr<kernel::JitKernelBase> kernel;
                        if (k.avx512) {
                            kernel = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx512_core>>(jcp);
                        } else {
                            kernel = std::make_shared<kernel::jit_rms_kernel<dnnl::impl::cpu::x64::avx2>>(jcp);
                        }
                        kernel->create_kernel();
                        return kernel;
                    })
                .first;
        };
        m_rms_kernel = get_kernel(ov::element::f32);
        if (m_gemm_precision != ov::element::f32) {
            m_rms_kernel_gemm = get_kernel(m_gemm_precision);
        }

        // Mix: out = (1/C) * sum_c gate_c * rbar_c, converted to the output precision, in one pass.
        const MixKernelKey mix_key{m_precision, D, C, avx512};
        m_mix_kernel =
            m_context->getRuntimeCache()
                ->getOrCreate(
                    mix_key,
                    [](const MixKernelKey& k) -> std::shared_ptr<kernel::JitKernelBase> {
                        kernel::jit_gated_residual_mix_compile_params jcp;
                        jcp.prc = k.precision;
                        jcp.data_size = k.data_size;
                        jcp.branches = k.branches;
                        jcp.scale = 1.0F / static_cast<float>(k.branches);
                        std::shared_ptr<kernel::JitKernelBase> kernel;
                        if (k.avx512) {
                            kernel = std::make_shared<
                                kernel::jit_gated_residual_mix_kernel<dnnl::impl::cpu::x64::avx512_core>>(jcp);
                        } else {
                            kernel =
                                std::make_shared<kernel::jit_gated_residual_mix_kernel<dnnl::impl::cpu::x64::avx2>>(
                                    jcp);
                        }
                        kernel->create_kernel();
                        return kernel;
                    })
                .first;
    }

    const size_t max_tile = maxTileRows();
    const size_t tile_rows = std::min(m_rows, max_tile);
    const size_t main_m = normalize_rows(tile_rows);
    const size_t rem_rows = m_rows > max_tile ? m_rows % max_tile : 0;
    const size_t rem_m = rem_rows != 0 ? normalize_rows(rem_rows) : 0;

    m_tile_rows = tile_rows;
    // Weights, scales and shapes are fixed for the executor's lifetime, so everything below only depends on the
    // two M values; nothing to do when they are unchanged (e.g. rows 100 and 110 both bucket to 112).
    if (m_main_m != 0 && main_m == m_main_m && rem_m == m_rem_m) {
        return true;
    }

    // The FC primitives take their scratchpad from the same shared block as the buffers below, starting at its
    // beginning, so those start after the largest of them.
    size_t fc_scratch = fcScratchBytes(main_m);
    if (rem_m != 0 && rem_m != main_m) {
        fc_scratch = std::max(fc_scratch, fcScratchBytes(rem_m));
    }

    if (main_m != m_main_m || fc_scratch != m_fc_scratch_bytes) {
        // One scratchpad block holding every per-tile buffer (sized for the main M), each region 64-byte aligned.
        const bool gemm_lowp = m_gemm_precision != ov::element::f32;
        m_fc_scratch_bytes = fc_scratch;
        size_t scratch_size = fc_scratch;
        const auto reserve = [&scratch_size](size_t bytes) {
            const size_t offset = scratch_size;
            scratch_size += (bytes + 63) / 64 * 64;
            return offset;
        };
        m_off_rbar = reserve(main_m * CD * sizeof(float));
        m_off_logit = reserve(main_m * CD * sizeof(float));
        m_off_t = reserve(main_m * L * m_gemm_precision.size());
        m_off_wg_gate = m_produce_write_gate ? reserve(main_m * C * m_precision.size()) : 0;
        m_off_rbar_gemm = gemm_lowp ? reserve(main_m * CD * sizeof(uint16_t)) : 0;
        const auto scratch_desc = BlockedDescCreator::getCommonCreators()
                                      .at(LayoutType::ncsp)
                                      ->createSharedDesc(ov::element::u8, Shape({scratch_size}));
        m_scratch = m_context->getScratchPad()->createScratchPadMem(scratch_desc);
    }

    // Uncompressed weights are consumed at the port precision, so the GEMM must run at exactly that precision;
    // supports() rejects configurations where the ISA can't (no upconvert path).
    OPENVINO_ASSERT(m_weights_compressed || m_precision == m_gemm_precision,
                    "GatedResidual uncompressed weights require the GEMM precision to match the port precision");

    m_main_m = main_m;
    m_rem_m = rem_m;

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

    // Re-derive the scratch pointers: the shared scratchpad block may have been reallocated (or reused
    // by other nodes) since the last call.
    auto* scratch = m_scratch->getDataAs<uint8_t>();
    m_rbar = reinterpret_cast<float*>(scratch + m_off_rbar);
    m_logit = reinterpret_cast<float*>(scratch + m_off_logit);
    m_t = scratch + m_off_t;
    m_wg_gate = m_produce_write_gate ? scratch + m_off_wg_gate : nullptr;
    m_rbar_gemm =
        m_gemm_precision != ov::element::f32 ? reinterpret_cast<uint16_t*>(scratch + m_off_rbar_gemm) : nullptr;

    // Inputs/outputs may carry any number of leading batch dims (e.g. [B,T,CD]); all row-indexed access below
    // assumes a flat [rows, *] layout, so they are collapsed into `rows`.
    auto* mixed_out = memory.at(gatedResidualDstArg(0))->getDataAs<uint8_t>();  // [rows, D] at the port precision

    // The RMS kernel reads the residual straight from the port memory in its native precision.
    const auto* residual_bytes = memory.at(gatedResidualSrcArg(ReadInputIdx::RESIDUAL))->getDataAs<const uint8_t>();
    const size_t residual_elem_size = m_precision.size();
    // The gain is read at the port precision too, and the kernel applies the "+1" (norm_gain_plus_one) itself.
    const auto* gain_bytes = memory.at(gatedResidualSrcArg(ReadInputIdx::NORM_WEIGHT))->getDataAs<const uint8_t>();

    const bool produce_write_gate = m_produce_write_gate;
    uint8_t* write_gate_out = nullptr;  // [rows, C] output port, contiguous ncsp at the port precision
    if (produce_write_gate) {
        write_gate_out = memory.at(gatedResidualDstArg(1))->getDataAs<uint8_t>();
    }

    const bool gemm_lowp = m_gemm_precision != ov::element::f32;
    const size_t tile_rows = m_tile_rows;

    // Process rows in tiles: full tiles run m_main_m rows through the GEMMs, the last partial tile m_rem_m. `cur_rows`
    // bounds the per-row elementwise steps and the stores to the real I/O tensors; a GEMM's M (`gemm_rows`) may be
    // larger (a bucketed M), in which case its extra source rows are zeroed and its extra results discarded.
    for (size_t tile_start = 0; tile_start < rows; tile_start += tile_rows) {
        const size_t cur_rows = std::min(tile_rows, rows - tile_start);
        const size_t gemm_rows = cur_rows == tile_rows ? m_main_m : m_rem_m;

        // Step 1: per-row RMS-norm through the shared JIT kernel, writing m_rbar (f32, the source for the later mix)
        // and, for a bf16/f16 GEMM, a copy at the GEMM precision into m_rbar_gemm. `row` indexes within the tile; the
        // residual input is read at the global row `tile_start + row`.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            float* rbar_row = m_rbar + row * CD;
            // Each branch is a D-element RMSNorm of the contiguous [rows, CD] residual with the gain slice
            // [c*D, (c+1)*D).
            const auto* src_row = residual_bytes + (tile_start + row) * CD * residual_elem_size;
            for (size_t c = 0; c < C; c++) {
                kernel::jit_rms_call_args args{src_row + c * D * residual_elem_size,
                                               reinterpret_cast<const float*>(gain_bytes + c * D * residual_elem_size),
                                               reinterpret_cast<uint8_t*>(rbar_row + c * D)};
                (*m_rms_kernel)(&args);
                if (gemm_lowp) {
                    // Same normalization again, stored straight as the GEMM's bf16/f16 source.
                    args.dst = reinterpret_cast<uint8_t*>(m_rbar_gemm + row * CD + c * D);
                    (*m_rms_kernel_gemm)(&args);
                }
            }
        });

        // The GEMM's bucketed M may exceed the real rows; zero the extra source rows (the shared scratchpad
        // holds arbitrary data from other nodes). Their results are never stored.
        if (cur_rows < gemm_rows) {
            std::fill(m_rbar + cur_rows * CD, m_rbar + gemm_rows * CD, 0.0F);
            if (gemm_lowp) {
                std::fill(m_rbar_gemm + cur_rows * CD, m_rbar_gemm + gemm_rows * CD, uint16_t{0});
            }
        }

        // Step 2: down-projection over the tile, T[M,L] = silu(Rbar[M,CD] @ w_down[L,CD]^T / C), stored at the GEMM
        // precision.
        void* const gemm_src = gemm_lowp ? static_cast<void*>(m_rbar_gemm) : static_cast<void*>(m_rbar);
        runProjection(m_proj[kDown], gemm_rows, gemm_src, m_t);

        // Step 3: up-projection over the tile, Gate[M,CD] = sigmoid(T[M,L] @ w_up[CD,L]^T) (logistic post-op).
        runProjection(m_proj[kUp], gemm_rows, m_t, m_logit);

        // Step 4: mix over C, one JIT kernel call per row: out = (1/C) * sum_c gate_c * rbar_c (m_logit holds the
        // gates), converted to the output precision as it is stored at the global row `tile_start + row`.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            kernel::jit_gated_residual_mix_call_args args{m_logit + row * CD,
                                                          m_rbar + row * CD,
                                                          mixed_out + (tile_start + row) * D * residual_elem_size};
            (*m_mix_kernel)(&args);
        });

        // Step 5: optional inject-projection over the tile,
        // WriteGate[M,C] = 2 * sigmoid(Rbar[M,CD] @ w_inject[C,CD]^T / C), stored at the port precision. The GEMM
        // writes the scratch gate buffer (a fixed destination, so the executor is not re-bound for every tile) and
        // only the valid rows are copied to the output.
        if (produce_write_gate) {
            runProjection(m_proj[kInject], gemm_rows, gemm_src, m_wg_gate);
            const size_t wg_elem_size = m_precision.size();
            std::memcpy(write_gate_out + tile_start * C * wg_elem_size, m_wg_gate, cur_rows * C * wg_elem_size);
        }
    }
}

void GatedResidualDnnlExecutor::execute_write(const MemoryArgs& memory) {
    const size_t C = m_hc_count;
    const size_t D = m_hidden_size;
    const size_t CD = C * D;

    // All ports are contiguous ncsp at the negotiated precision, with any leading batch dims flattened into
    // `rows`: residual/out are [rows, C*D], sublayer_y is [rows, D], write_gate is [rows, C].
    const size_t rows = shape_size(memory.at(gatedResidualSrcArg(WriteInputIdx::RESIDUAL))->getStaticDims()) / CD;
    const auto* residual = memory.at(gatedResidualSrcArg(WriteInputIdx::RESIDUAL))->getDataAs<const uint8_t>();
    const auto* y = memory.at(gatedResidualSrcArg(WriteInputIdx::SUBLAYER_Y))->getDataAs<const uint8_t>();
    auto* out = memory.at(gatedResidualDstArg(0))->getDataAs<uint8_t>();
    const auto* gate = memory.at(gatedResidualSrcArg(WriteInputIdx::WRITE_GATE))->getDataAs<const uint8_t>();
    const size_t elem_size = m_precision.size();

    // One kernel call per (row, branch): out = residual + gate * y over D elements (the kernel loads the gate
    // itself), in f32 with a single rounding on store; no scratch buffers are needed.
    m_context->getCpuParallel()->parallel_for2d(rows, C, [&](size_t row, size_t c) {
        const size_t branch_offset = (row * CD + c * D) * elem_size;
        kernel::jit_gated_residual_write_call_args args{residual + branch_offset,
                                                        y + row * D * elem_size,
                                                        gate + (row * C + c) * elem_size,
                                                        out + branch_offset};
        (*m_write_kernel)(&args);
    });
}

}  // namespace ov::intel_cpu

#endif  // OPENVINO_ARCH_X86_64
