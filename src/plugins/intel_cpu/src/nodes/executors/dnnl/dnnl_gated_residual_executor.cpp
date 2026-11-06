// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#    include <oneapi/dnnl/dnnl_types.h>

#    include <algorithm>
#    include <common/primitive_hashing.hpp>
#    include <common/utils.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <functional>
#    include <memory>
#    include <numeric>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_common.hpp>
#    include <unordered_map>
#    include <vector>

#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "cpu_memory.h"
#    include "cpu_shape.h"
#    include "dnnl_extension_utils.h"
#    include "memory_desc/cpu_memory_desc.h"
#    include "nodes/common/blocked_desc_creator.h"
#    include "nodes/executors/dnnl/dnnl_gated_residual_executor.hpp"
#    include "nodes/executors/dnnl/dnnl_utils.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/gated_residual_config.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "nodes/kernels/x64/gated_residual_mix_kernel.hpp"
#    include "nodes/kernels/x64/gated_residual_write_kernel.hpp"
#    include "nodes/kernels/x64/rms_kernel.hpp"
#    include "onednn/iml_type_mapper.h"
#    include "openvino/core/except.hpp"
#    include "openvino/core/shape.hpp"
#    include "openvino/core/type/bfloat16.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/core/type/float16.hpp"
#    include "openvino/op/gated_residual.hpp"
#    include "openvino/runtime/system_conf.hpp"
#    include "thread_pool_imp.hpp"
#    include "utils/general_utils.h"
#    include "utils/plain_tensor.hpp"

namespace ov::intel_cpu {

namespace {

using ReadInputIdx = ov::op::internal::GatedResidual::ReadInputIdx;
using WriteInputIdx = ov::op::internal::GatedResidual::WriteInputIdx;
using Mode = ov::op::internal::GatedResidual::Mode;

// Memory descriptors for a single `dst[rows,N] = lhs[rows,K] @ weights[N,K]^T` batched matmul.
// `weights` is expressed with its logical (K,N) shape via strides over its (N,K) row-major storage
// (no data copy), following the rhsShape/rhsStride technique in nodes/interaction.cpp.
struct ProjectionDescs {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc dst_md;
};

ProjectionDescs make_projection_descs(size_t rows,
                                      size_t k,
                                      size_t n,
                                      dnnl::memory::data_type src_dt,
                                      dnnl::memory::data_type weights_dt,
                                      dnnl::memory::data_type dst_dt) {
    using dnnl::memory;
    const auto r = static_cast<int64_t>(rows);
    const auto kk = static_cast<int64_t>(k);
    const auto nn = static_cast<int64_t>(n);
    return ProjectionDescs{memory::desc({r, kk}, src_dt, memory::dims{kk, 1}),
                           memory::desc({kk, nn}, weights_dt, memory::dims{1, kk}),
                           memory::desc({r, nn}, dst_dt, memory::dims{nn, 1})};
}

template <typename Dims>
dnnl::memory::dims to_dnnl_dims(const Dims& dims) {
    dnnl::memory::dims out(dims.size());
    std::transform(dims.begin(), dims.end(), out.begin(), [](auto d) {
        return static_cast<int64_t>(d);
    });
    return out;
}

// Mirrors DnnlPostOpsComposer::getGroupDims (src/plugins/intel_cpu/src/dnnl_postops_composer.cpp),
// duplicated locally since that helper is a file-local `static` function. `wei_dims`/`scale_dims`
// are the weight's [N,K] shape and the decompression scale/zero-point's logical [N,G] shape (the graph port is
// [G,N]; callers swap its dims).
dnnl::memory::dims decompression_group_dims(const dnnl::memory::dims& wei_dims, const dnnl::memory::dims& scale_dims) {
    if (scale_dims[0] == 1 && scale_dims[1] == 1) {
        return {};
    }
    const auto n = wei_dims[0];
    const auto k = wei_dims[1];
    return {k / scale_dims[1], n / scale_dims[0]};
}

// Mirrors DnnlPostOpsComposer::getMask, duplicated for the same reason as decompression_group_dims.
int decompression_mask(const dnnl::memory::dims& wei_dims, const dnnl::memory::dims& group_dims) {
    constexpr int mask_n = 1 << 1;
    constexpr int mask_k = 1 << 0;
    int mask = 0;
    if (!group_dims.empty() && group_dims[1] != wei_dims[0]) {
        mask += mask_n;
    }
    if (!group_dims.empty() && group_dims[0] != wei_dims[1]) {
        mask += mask_k;
    }
    return mask;
}

// s32 zero-points work uniformly across all supported weight dtypes (u8/s8/u4/s4), so there's no
// need to special-case the zero-point storage type per weight dtype family.
dnnl::memory::data_type decompression_zero_point_dt(dnnl::memory::data_type weights_dt) {
    (void)weights_dt;
    return dnnl::memory::data_type::s32;
}

// Builds the oneDNN native weight-decompression primitive_attr for a compressed projection. Scales are always f32
// (the graph port's f16/bf16/f32 scales are converted once). Zero-points (when present) are s32: oneDNN's
// weights-decompression extension takes integer zero-points, and s32 works for every supported weight type; the
// graph port's u4/i4/u8/i8 zero-points are widened once in make_decompression_zero_point_memory().
dnnl::primitive_attr build_decompression_attr(const dnnl::memory::dims& wei_dims,
                                              const dnnl::memory::dims& scale_dims,
                                              dnnl::memory::data_type weights_dt,
                                              bool has_zero_points) {
    dnnl::primitive_attr attr;
    const auto group_dims = decompression_group_dims(wei_dims, scale_dims);
    const auto mask = decompression_mask(wei_dims, group_dims);
    attr.set_scales(DNNL_ARG_WEIGHTS, mask, group_dims, dnnl::memory::data_type::f32);
    if (has_zero_points) {
        attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, group_dims, decompression_zero_point_dt(weights_dt));
    }
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);
    return attr;
}

size_t element_count(const MemoryPtr& mem) {
    const auto& dims = mem->getStaticDims();
    return std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
}

// Wraps a rank-2 decompression scale port -- shaped [G, N] and stored row-major that way (N contiguous), which is
// already the groups-outer physical order oneDNN's weights-decompression scale argument expects (element (n, g) at
// offset g * N + n) -- in a dnnl memory view with the logical (N, G) shape. The scales are floating point on the
// graph port (f16 / bf16 / f32); `scratch` holds the f32 copy oneDNN reads.
dnnl::memory make_decompression_scale_memory(const MemoryPtr& mem,
                                             const dnnl::engine& engine,
                                             std::vector<float>& scratch) {
    const auto dims = to_dnnl_dims(mem->getStaticDims());
    const auto groups = dims[0];
    const auto n = dims[1];
    const auto count = element_count(mem);
    scratch.resize(count);
    const auto prc = mem->getPrecision();
    if (prc == ov::element::f32) {
        std::copy_n(mem->getDataAs<const float>(), count, scratch.begin());
    } else if (prc == ov::element::f16) {
        const auto* src = mem->getDataAs<const ov::float16>();
        std::transform(src, src + count, scratch.begin(), [](ov::float16 v) {
            return static_cast<float>(v);
        });
    } else if (prc == ov::element::bf16) {
        const auto* src = mem->getDataAs<const ov::bfloat16>();
        std::transform(src, src + count, scratch.begin(), [](ov::bfloat16 v) {
            return static_cast<float>(v);
        });
    } else {
        OPENVINO_THROW("GatedResidual: unsupported decompression scale precision ", prc);
    }
    const dnnl::memory::desc md({n, groups}, dnnl::memory::data_type::f32, dnnl::memory::dims{1, n});
    return {md, engine, scratch.data()};
}

// Same logical (N, G) shape / groups-outer physical layout as make_decompression_scale_memory(). The zero-point port
// has the weights' type (u4 / i4 packed two per byte, low nibble first, or u8 / i8); `scratch` holds the s32 copy.
dnnl::memory make_decompression_zero_point_memory(const MemoryPtr& mem,
                                                  const dnnl::engine& engine,
                                                  std::vector<int32_t>& scratch) {
    const auto dims = to_dnnl_dims(mem->getStaticDims());
    const auto groups = dims[0];
    const auto n = dims[1];
    const auto count = element_count(mem);
    scratch.resize(count);
    const auto* src = mem->getDataAs<const uint8_t>();
    const auto prc = mem->getPrecision();
    if (prc == ov::element::u8) {
        std::copy_n(src, count, scratch.begin());
    } else if (prc == ov::element::i8) {
        std::transform(src, src + count, scratch.begin(), [](uint8_t v) {
            return static_cast<int32_t>(static_cast<int8_t>(v));
        });
    } else if (prc == ov::element::u4 || prc == ov::element::i4) {
        const bool is_signed = prc == ov::element::i4;
        for (size_t i = 0; i < count; i++) {
            const uint8_t nibble = (i % 2 == 0) ? (src[i / 2] & 0x0FU) : (src[i / 2] >> 4U);
            scratch[i] = is_signed ? static_cast<int32_t>(nibble) - ((nibble & 0x8U) != 0U ? 16 : 0)
                                   : static_cast<int32_t>(nibble);
        }
    } else {
        OPENVINO_THROW("GatedResidual: unsupported decompression zero-point precision ", prc);
    }
    const dnnl::memory::desc md({n, groups}, dnnl::memory::data_type::s32, dnnl::memory::dims{1, n});
    return {md, engine, scratch.data()};
}

// Rounds a row count up to a bucket, bounding the number of distinct GEMM shapes (and so cached primitives
// and JIT builds) a dynamic row count can produce -- the same rule GatherMatmul uses for its per-expert M.
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

// Runtime-cache key/value for the projection primitives. The cache is shared across nodes in a stream,
// so every GatedResidual layer with the same projection shapes (all of them, in a real model) reuses one
// primitive per projection instead of each building its own, and revisiting a row-tile size (dynamic
// shapes) is a cache hit rather than a primitive_desc + JIT rebuild.
// Elementwise work fused into a projection's GEMM epilogue (oneDNN post-ops, applied to the f32 accumulator
// before the store converts to the destination type).
enum class MatmulEpilogue : uint8_t {
    None,
    ScaledSilu,  // silu(x * scale) = swish(linear(x, scale)), the low-rank activation
    Logistic,    // sigmoid(x), the gates
    WriteGate,   // 2 * sigmoid(x * scale), the write gate
};

struct MatmulKey {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::dims scale_dims;  // empty for uncompressed weights
    bool compressed = false;
    bool has_zero_points = false;
    MatmulEpilogue epilogue = MatmulEpilogue::None;
    float scale = 1.0F;  // inv_c for ScaledSilu

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(*src_md.get()));
        seed = hash_combine(seed, get_md_hash(*weights_md.get()));
        seed = hash_combine(seed, get_md_hash(*dst_md.get()));
        for (const auto d : scale_dims) {
            seed = hash_combine(seed, static_cast<size_t>(d));
        }
        seed = hash_combine(seed, static_cast<size_t>(compressed));
        seed = hash_combine(seed, static_cast<size_t>(has_zero_points));
        seed = hash_combine(seed, static_cast<size_t>(epilogue));
        seed = hash_combine(seed, std::hash<float>{}(scale));
        return seed;
    }

    bool operator==(const MatmulKey& rhs) const {
        return src_md == rhs.src_md && weights_md == rhs.weights_md && dst_md == rhs.dst_md &&
               scale_dims == rhs.scale_dims && compressed == rhs.compressed && has_zero_points == rhs.has_zero_points &&
               epilogue == rhs.epilogue && scale == rhs.scale;
    }
};

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

struct CachedMatmul {
    dnnl::matmul prim;
    dnnl::memory::desc weights_md;  // the weights layout the primitive expects (blocked when the key said "any")
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
    // READ/FINAL_MIX run the RMSNorm JIT kernel, which needs AVX2; WRITE has no RMSNorm.
    if (config.attrs.mode != ov::op::internal::GatedResidual::Mode::WRITE && !mayiuse(cpu_isa_t::avx2)) {
        return false;
    }
    // Uncompressed weights are fed to the GEMM at the port precision (no upconvert path), so a
    // bf16/f16 port needs native bf16/f16 matmul support. Compressed weights are decompressed by
    // oneDNN and may still run the GEMM in f32.
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
      m_stream(make_stream(context->getEngine(), context->getThreadPool())),
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
    if (m_mode != Mode::WRITE && !m_weights_compressed) {
        // Every projection runs from a prepacked copy of its (constant) weights. Measured end to end on the
        // production shape, packing all three beats packing only some of them, in prefill and in decode.
        m_packed[kDown].enabled = true;
        m_packed[kUp].enabled = true;
        m_packed[kInject].enabled = m_produce_write_gate;
    }
    m_precision = memory.at(gatedResidualSrcArg(ReadInputIdx::RESIDUAL))->getPrecision();

    // The down/up/inject weight projections run through dnnl::matmul. Use bf16/f16 as the GEMM's
    // compute precision (src+weights; dst/accumulation always stays f32) only when the port
    // precision is bf16/f16 and the CPU ISA accelerates matmul at that precision; otherwise the
    // GEMM upconverts to f32, same idiom as Interaction::initSupportedPrimitiveDescriptors().
    using dnnl::impl::cpu::x64::cpu_isa_t;
    using dnnl::impl::cpu::x64::mayiuse;
    if (m_precision == ov::element::bf16 && mayiuse(cpu_isa_t::avx512_core_bf16)) {
        m_gemm_precision = ov::element::bf16;
    } else if (m_precision == ov::element::f16 && mayiuse(cpu_isa_t::avx512_core_fp16)) {
        m_gemm_precision = ov::element::f16;
    }
}

impl_desc_type GatedResidualDnnlExecutor::implType() const {
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
                            // supports() does not gate WRITE on AVX2 (it has no RMSNorm), so fall back to
                            // the narrowest ISA that always exists on x86-64 targets of this plugin.
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
        // The kernels only depend on (precision, dst precision, D, eps, ISA), so every GatedResidual layer
        // with the same hidden size shares them through the stream's runtime cache (as RMSNorm does)
        // instead of each compiling its own. Two instances run per branch when the GEMM is bf16/f16: one
        // writes the f32 Rbar (the mix reads it), the other writes the GEMM's bf16/f16 source directly, so
        // there is no separate downcast pass and both values come from the identical f32 computation.
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

    // Rows run in tiles; each tile re-streams the weights, so bigger tiles are faster. Tile size is capped by a
    // scratch budget in the shared scratchpad. GEMM M is bucketed (normalize_rows); padded rows are zeroed and never
    // stored.
    constexpr size_t kTileRowAlignment = 32;  // keeps every normalized remainder <= the tile
    constexpr size_t kMinTileRows = 32;
    constexpr size_t kMaxTileRows = 512;
    constexpr size_t kScratchBudgetBytes = 64UL << 20;
    const size_t bytes_per_row = (2 * CD + L) * sizeof(float);
    size_t max_tile = std::clamp(kScratchBudgetBytes / bytes_per_row, kMinTileRows, kMaxTileRows);
    max_tile -= max_tile % kTileRowAlignment;
    const size_t tile_rows = std::min(m_rows, max_tile);
    const size_t main_m = normalize_rows(tile_rows);
    const size_t rem_rows = m_rows > max_tile ? m_rows % max_tile : 0;
    const size_t rem_m = rem_rows != 0 ? normalize_rows(rem_rows) : 0;

    m_tile_rows = tile_rows;
    // Weights, scales and shapes are fixed for the executor's lifetime, so the GEMMs only depend on the
    // two M values; nothing to do when they are unchanged (e.g. rows 100 and 110 both bucket to 112).
    if (m_main.m != 0 && main_m == m_main.m && rem_m == m_rem_m) {
        return true;
    }

    if (main_m != m_main.m) {
        // One scratchpad block holding every per-tile buffer (sized for the main M), each region 64-byte aligned.
        const bool gemm_lowp = m_gemm_precision != ov::element::f32;
        size_t scratch_size = 0;
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

    // Uncompressed weights are consumed zero-copy at the port precision, so the GEMM must run at
    // exactly that precision; supports() rejects configurations where the ISA can't (no upconvert path).
    OPENVINO_ASSERT(m_weights_compressed || m_precision == m_gemm_precision,
                    "GatedResidual uncompressed weights require the GEMM precision to match the port precision");

    m_main = createTileGemms(main_m, memory);
    m_rem = TileGemms{};
    if (rem_m != 0) {
        m_rem = rem_m == main_m ? m_main : createTileGemms(rem_m, memory);
    }
    m_rem_m = rem_m;

    const size_t cd = m_hc_count * m_hidden_size;
    const auto gemm_dt = DnnlExtensionUtils::ElementTypeToDataType(m_gemm_precision);
    const auto f32_dt = DnnlExtensionUtils::ElementTypeToDataType(ov::element::f32);
    // plain (K, N) view of each [N, K] row-major weight, and the port that holds it
    struct ProjectionInfo {
        size_t k;
        size_t n;
        size_t port;
    };
    const ProjectionInfo infos[3] = {{cd, m_lowrank, ReadInputIdx::W_DOWN},
                                     {m_lowrank, cd, ReadInputIdx::W_UP},
                                     {cd, m_hc_count, m_layout.w_inject}};
    for (size_t i = 0; i < 3; i++) {
        auto& packed = m_packed[i];
        if (!packed.enabled || !packed.ready || packed.data) {
            continue;
        }
        // The weights are graph constants: reorder once, through the shared weights cache (identified by the
        // weights' address and the target layout), into the layout the GEMM wants.
        const auto plain_md = make_projection_descs(1, infos[i].k, infos[i].n, gemm_dt, gemm_dt, f32_dt).weights_md;
        packed.data = utils::prepareWeightsMemory(DnnlExtensionUtils::makeDescriptor(plain_md),
                                                  DnnlExtensionUtils::makeDescriptor(packed.md),
                                                  memory.at(gatedResidualSrcArg(infos[i].port)),
                                                  m_context->getEngine(),
                                                  m_context->getRuntimeCache(),
                                                  m_context->getWeightsCache(),
                                                  nullptr,  // keyed by layout only: unsafe for several weights per node
                                                  m_context->getThreadPool());
        packed.view = dnnl::memory(packed.md, m_context->getEngine(), packed.data->getData());
    }
    return true;
}

GatedResidualDnnlExecutor::TileGemms GatedResidualDnnlExecutor::createTileGemms(size_t m, const MemoryArgs& memory) {
    const size_t C = m_hc_count;
    const size_t L = m_lowrank;
    const size_t CD = C * m_hidden_size;
    const auto src_dt = DnnlExtensionUtils::ElementTypeToDataType(m_gemm_precision);
    const auto dst_dt = DnnlExtensionUtils::ElementTypeToDataType(ov::element::f32);

    // Builds one projection's GEMM for M = m through the runtime cache. For compressed weights,
    // `weight_port` carries the port's native u8/i8/u4/i4 element type into the matmul's weights_md, and
    // `scale_port`'s static shape (vs. the weight's own [n,k] shape) determines the decompression
    // groupDims/mask baked into the primitive_attr -- see build_decompression_attr().
    const auto build = [&](size_t k,
                           size_t n,
                           size_t weight_port,
                           size_t scale_port,
                           dnnl::memory::data_type out_dt,
                           MatmulEpilogue epilogue = MatmulEpilogue::None,
                           float scale = 1.0F,
                           int packed_idx = -1) {
        const auto weights_dt =
            m_weights_compressed
                ? DnnlExtensionUtils::ElementTypeToDataType(memory.at(gatedResidualSrcArg(weight_port))->getPrecision())
                : src_dt;
        const auto descs = make_projection_descs(m, k, n, src_dt, weights_dt, out_dt);
        // Prepacked weights: ask oneDNN for its preferred (blocked) layout the first time; every later M builds on
        // that same layout (the one packed copy serves all M), so the packed weights never depend on the row count.
        const bool pack_weights = packed_idx >= 0 && m_packed[packed_idx].enabled;
        const auto weights_key_md = [&]() -> dnnl::memory::desc {
            if (!pack_weights) {
                return descs.weights_md;
            }
            if (m_packed[packed_idx].ready) {
                return m_packed[packed_idx].md;
            }
            return {{static_cast<int64_t>(k), static_cast<int64_t>(n)}, weights_dt, dnnl::memory::format_tag::any};
        }();
        MatmulKey key{descs.src_md,
                      weights_key_md,
                      descs.dst_md,
                      {},
                      m_weights_compressed,
                      m_weights_have_zero_points,
                      epilogue,
                      scale};
        if (m_weights_compressed) {
            // The scale port is [G, N]; the group-dims helpers take the logical [N, G] shape.
            const auto port_dims = to_dnnl_dims(memory.at(gatedResidualSrcArg(scale_port))->getStaticDims());
            key.scale_dims = {port_dims[1], port_dims[0]};
        }
        const auto& engine = m_context->getEngine();
        const auto result = m_context->getRuntimeCache()->getOrCreate(key, [&engine](const MatmulKey& k) {
            dnnl::primitive_attr attr;
            if (k.compressed) {
                // weights_md is the logical (K, N) view; the decompression group dims/mask use the [N, K] shape.
                const auto wei_dims = k.weights_md.get_dims();
                const dnnl::memory::dims nk{wei_dims[1], wei_dims[0]};
                attr = build_decompression_attr(nk, k.scale_dims, k.weights_md.get_data_type(), k.has_zero_points);
            }
            if (k.epilogue != MatmulEpilogue::None) {
                dnnl::post_ops ops;
                if (k.epilogue == MatmulEpilogue::ScaledSilu) {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, k.scale, 0.0F);
                    ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0F, 0.0F);  // x * sigmoid(x)
                } else if (k.epilogue == MatmulEpilogue::WriteGate) {
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, k.scale, 0.0F);
                    ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 1.0F, 0.0F);
                    ops.append_eltwise(dnnl::algorithm::eltwise_linear, 2.0F, 0.0F);
                } else {
                    ops.append_eltwise(dnnl::algorithm::eltwise_logistic, 1.0F, 0.0F);
                }
                attr.set_post_ops(ops);
            }
            const dnnl::matmul::primitive_desc pd(engine, k.src_md, k.weights_md, k.dst_md, attr);
            return std::make_shared<CachedMatmul>(CachedMatmul{dnnl::matmul(pd), pd.weights_desc()});
        });
        // dnnl::matmul is a ref-counted handle, so this copy stays valid even if the cache evicts the entry.
        return ProjectionGemm{result.first->prim,
                              descs.src_md,
                              pack_weights ? result.first->weights_md : descs.weights_md,
                              descs.dst_md};
    };

    TileGemms gemms;
    gemms.m = m;
    // The down GEMM produces T = silu(down(Rbar) / C) directly, stored at the GEMM precision so the up GEMM
    // reads it as-is (no separate activation pass, no downcast, no f32 T buffer).
    gemms.down = build(CD,
                       L,
                       ReadInputIdx::W_DOWN,
                       ReadInputIdx::SCALE_DOWN,
                       src_dt,
                       MatmulEpilogue::ScaledSilu,
                       1.0F / static_cast<float>(C),
                       kDown);
    // The up GEMM produces the gates directly: logit -> sigmoid in the epilogue.
    gemms.up = build(L, CD, ReadInputIdx::W_UP, ReadInputIdx::SCALE_UP, dst_dt, MatmulEpilogue::Logistic, 1.0F, kUp);
    if (m_produce_write_gate) {
        // The write gate 2 * sigmoid(logit / C) is the GEMM epilogue, stored at the output port's precision so
        // the valid rows can be copied to the output as-is.
        gemms.inject = build(CD,
                             C,
                             m_layout.w_inject,
                             m_layout.scale_inject,
                             DnnlExtensionUtils::ElementTypeToDataType(m_precision),
                             MatmulEpilogue::WriteGate,
                             1.0F / static_cast<float>(C),
                             kInject);
    }
    // Remember the layout each packed projection settled on; every other M builds on it.
    const ProjectionGemm* projection_gemms[3] = {&gemms.down, &gemms.up, &gemms.inject};
    for (size_t i = 0; i < 3; i++) {
        if (m_packed[i].enabled && !m_packed[i].ready && projection_gemms[i]->prim) {
            m_packed[i].md = projection_gemms[i]->weights_md;
            m_packed[i].ready = true;
        }
    }
    return gemms;
}

void GatedResidualDnnlExecutor::execute_read_or_final_mix(const MemoryArgs& memory) {
    const auto& strm = m_stream;

    // Re-derive the scratch pointers: the shared scratchpad block may have been reallocated (or reused
    // by other nodes) since the last call.
    auto* scratch = m_scratch->getDataAs<uint8_t>();
    m_rbar = reinterpret_cast<float*>(scratch + m_off_rbar);
    m_logit = reinterpret_cast<float*>(scratch + m_off_logit);
    m_t = scratch + m_off_t;
    m_wg_gate = m_produce_write_gate ? scratch + m_off_wg_gate : nullptr;
    m_rbar_gemm =
        m_gemm_precision != ov::element::f32 ? reinterpret_cast<uint16_t*>(scratch + m_off_rbar_gemm) : nullptr;
    const size_t C = m_hc_count;
    const size_t D = m_hidden_size;
    const size_t CD = C * D;
    const size_t rows = m_rows;
    if (rows == 0) {
        return;
    }

    // Inputs/outputs may carry any number of leading batch dims (e.g. [B,T,CD]); every row-indexed
    // row-indexed access below assumes a flat [rows, *] stride, so collapse the leading
    // dims into `rows` here rather than letting ptr<T>(row) use the un-flattened dim-0 stride.
    PlainTensor w_down(memory.at(gatedResidualSrcArg(ReadInputIdx::W_DOWN)));
    PlainTensor w_up(memory.at(gatedResidualSrcArg(ReadInputIdx::W_UP)));
    auto* mixed_out = memory.at(gatedResidualDstArg(0))->getDataAs<uint8_t>();  // [rows, D] at the port precision

    // The RMS kernel reads the residual (and the gain) in their native precision straight from the port memory.
    const auto* residual_bytes = memory.at(gatedResidualSrcArg(ReadInputIdx::RESIDUAL))->getDataAs<const uint8_t>();
    const size_t residual_elem_size = m_precision.size();
    // The gain is passed to the RMS kernel as-is: it loads it at the port precision and applies the "+1"
    // (norm_gain_plus_one) itself, so there is no conversion pass or f32 copy.
    const auto* gain_bytes = memory.at(gatedResidualSrcArg(ReadInputIdx::NORM_WEIGHT))->getDataAs<const uint8_t>();

    const bool produce_write_gate = m_produce_write_gate;
    PlainTensor w_inject;
    uint8_t* write_gate_out = nullptr;  // [rows, C] output port, contiguous ncsp at the port precision
    if (produce_write_gate) {
        w_inject.reset(memory.at(gatedResidualSrcArg(m_layout.w_inject)));
        write_gate_out = memory.at(gatedResidualDstArg(1))->getDataAs<uint8_t>();
    }

    const bool compressed = m_weights_compressed;
    const auto& engine = m_context->getEngine();
    const bool gemm_lowp = m_gemm_precision != ov::element::f32;
    const size_t tile_rows = m_tile_rows;

    // Weight/decompression views are tile-invariant (weights_md only depends on k/n, not M, and the
    // scale/zero-point tensors are whole-tensor constants), so they're built once here instead of once
    // per tile, from the main GEMMs' descriptors.
    // Packed projections use the copy prepared in update(); the others view the plain graph weights.
    dnnl::memory down_plain_mem;
    dnnl::memory up_plain_mem;
    dnnl::memory inject_plain_mem;
    if (!m_packed[kDown].enabled) {
        down_plain_mem = dnnl::memory(m_main.down.weights_md, engine, w_down.ptr_v(0));
    }
    if (!m_packed[kUp].enabled) {
        up_plain_mem = dnnl::memory(m_main.up.weights_md, engine, w_up.ptr_v(0));
    }
    if (produce_write_gate && !m_packed[kInject].enabled) {
        inject_plain_mem = dnnl::memory(m_main.inject.weights_md, engine, w_inject.ptr_v(0));
    }
    const dnnl::memory& down_weights_mem = m_packed[kDown].enabled ? m_packed[kDown].view : down_plain_mem;
    const dnnl::memory& up_weights_mem = m_packed[kUp].enabled ? m_packed[kUp].view : up_plain_mem;
    const dnnl::memory& inject_weights_mem = m_packed[kInject].enabled ? m_packed[kInject].view : inject_plain_mem;

    // The decompression views need an f32 (scales) / s32 (zero-points) copy of each
    // scale/zero-point tensor. They are graph constants (enforced by the node), so that is done once.
    if (compressed && !m_decomp_ready) {
        const auto port = [&memory](size_t idx) -> const MemoryPtr& {
            return memory.at(gatedResidualSrcArg(idx));
        };
        m_down_scale_mem = make_decompression_scale_memory(port(ReadInputIdx::SCALE_DOWN), engine, m_w_down_scale_f32);
        m_up_scale_mem = make_decompression_scale_memory(port(ReadInputIdx::SCALE_UP), engine, m_w_up_scale_f32);
        if (m_weights_have_zero_points) {
            m_down_zp_mem = make_decompression_zero_point_memory(port(ReadInputIdx::ZP_DOWN), engine, m_w_down_zp);
            m_up_zp_mem = make_decompression_zero_point_memory(port(ReadInputIdx::ZP_UP), engine, m_w_up_zp);
        }
        if (produce_write_gate) {
            m_inject_scale_mem =
                make_decompression_scale_memory(port(m_layout.scale_inject), engine, m_w_inject_scale_f32);
            if (m_weights_have_zero_points) {
                m_inject_zp_mem = make_decompression_zero_point_memory(port(m_layout.zp_inject), engine, m_w_inject_zp);
            }
        }
        m_decomp_ready = true;
    }

    // Process rows in tiles (see update() for the tile size and the dynamic-M GEMM selection): full tiles
    // run the main GEMMs, the last partial tile runs the remainder GEMMs. `cur_rows` bounds the per-row
    // elementwise steps and the stores back to the real I/O tensors; a GEMM's M (`gemm_rows`) may be larger
    // (a bucketed M), in which case its extra source rows are zeroed and its extra results discarded.
    for (size_t tile_start = 0; tile_start < rows; tile_start += tile_rows) {
        const size_t cur_rows = std::min(tile_rows, rows - tile_start);
        const TileGemms& g = cur_rows == tile_rows ? m_main : m_rem;
        const size_t gemm_rows = g.m;

        // Step 1: per-row RMS-norm through the shared JIT kernel (which reads the residual in its
        // native precision), writing m_rbar (f32, always the source of truth for the later gate/mix
        // reduce) and, when the GEMM runs in bf16/f16, a downcast copy into m_rbar_gemm. `row`
        // indexes within the tile (scratch buffers are tile_rows-sized); the real residual input is
        // read at the global row `tile_start + row`.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            float* rbar_row = m_rbar + row * CD;
            // The residual port is contiguous ncsp [rows, CD] at m_precision; each branch is a
            // D-element RMSNorm with the gain slice [c*D, (c+1)*D), writing f32 (and, for a
            // bf16/f16 GEMM, the converted copy from a second kernel instance).
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

        // Step 2: down-projection over the tile, T[M,L] = silu(Rbar[M,CD] @ w_down[L,CD]^T / C) (swish post-ops),
        // stored at the GEMM precision.
        {
            void* src_ptr = gemm_lowp ? static_cast<void*>(m_rbar_gemm) : static_cast<void*>(m_rbar);
            dnnl::memory src_mem(g.down.src_md, engine, src_ptr);
            dnnl::memory dst_mem(g.down.dst_md, engine, m_t);
            std::unordered_map<int, dnnl::memory> mem_ags{{DNNL_ARG_SRC, src_mem},
                                                          {DNNL_ARG_WEIGHTS, down_weights_mem},
                                                          {DNNL_ARG_DST, dst_mem}};
            if (compressed) {
                mem_ags.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_down_scale_mem);
                if (m_weights_have_zero_points) {
                    mem_ags.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_down_zp_mem);
                }
            }
            g.down.prim.execute(strm, mem_ags);
        }

        // Step 3: up-projection over the tile, Gate[M,CD] = sigmoid(T[M,L] @ w_up[CD,L]^T) (logistic post-op).
        {
            void* src_ptr = m_t;
            dnnl::memory src_mem(g.up.src_md, engine, src_ptr);
            dnnl::memory dst_mem(g.up.dst_md, engine, m_logit);
            std::unordered_map<int, dnnl::memory> mem_ags{{DNNL_ARG_SRC, src_mem},
                                                          {DNNL_ARG_WEIGHTS, up_weights_mem},
                                                          {DNNL_ARG_DST, dst_mem}};
            if (compressed) {
                mem_ags.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_up_scale_mem);
                if (m_weights_have_zero_points) {
                    mem_ags.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_up_zp_mem);
                }
            }
            g.up.prim.execute(strm, mem_ags);
        }

        // Step 4: mix over C, one JIT kernel call per row: out = (1/C) * sum_c gate_c * rbar_c (m_logit holds the
        // gates, sigmoid fused into the up GEMM), converted to the output precision as it is stored. The real
        // output tensor is written at the global row `tile_start + row`.
        m_context->getCpuParallel()->parallel_for(cur_rows, [&](size_t row) {
            kernel::jit_gated_residual_mix_call_args args{m_logit + row * CD,
                                                          m_rbar + row * CD,
                                                          mixed_out + (tile_start + row) * D * residual_elem_size};
            (*m_mix_kernel)(&args);
        });

        // Step 5: optional inject-projection over the tile, WriteGate[M,C] = 2 * sigmoid(Rbar[M,CD] @
        // w_inject[C,CD]^T / C) (linear/logistic/linear post-ops), stored at the port precision.
        if (produce_write_gate) {
            void* src_ptr = gemm_lowp ? static_cast<void*>(m_rbar_gemm) : static_cast<void*>(m_rbar);
            dnnl::memory src_mem(g.inject.src_md, engine, src_ptr);
            // With no padded rows the GEMM writes the output tensor directly; otherwise (bucketed M > real rows)
            // it writes scratch and only the valid rows are copied, so the padding never overruns the output.
            const size_t wg_elem_size = m_precision.size();
            const bool wg_direct = cur_rows == gemm_rows;
            uint8_t* wg_dst = wg_direct ? write_gate_out + tile_start * C * wg_elem_size : m_wg_gate;
            dnnl::memory dst_mem(g.inject.dst_md, engine, wg_dst);
            std::unordered_map<int, dnnl::memory> mem_ags{{DNNL_ARG_SRC, src_mem},
                                                          {DNNL_ARG_WEIGHTS, inject_weights_mem},
                                                          {DNNL_ARG_DST, dst_mem}};
            if (compressed) {
                mem_ags.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m_inject_scale_mem);
                if (m_weights_have_zero_points) {
                    mem_ags.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, m_inject_zp_mem);
                }
            }
            g.inject.prim.execute(strm, mem_ags);

            if (!wg_direct) {
                std::memcpy(write_gate_out + tile_start * C * wg_elem_size, m_wg_gate, cur_rows * C * wg_elem_size);
            }
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
