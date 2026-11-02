// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cache/multi_cache.h>

#include <common/utils.hpp>
#include <cstddef>
#include <memory>

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}  // namespace ov::intel_cpu::kernel

namespace ov::intel_cpu::node::qsa {

// What the JIT kernels of the RMSNorm and the RoPE nodes are specialized for; also the key in the params cache.
struct RmsRopeKey {
    size_t dim = 0;         // channels of a row
    size_t rotary_dim = 0;  // the channels rotated by RoPE
    float eps = 0.0F;

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, dim);
        seed = dnnl::impl::hash_combine(seed, rotary_dim);
        return dnnl::impl::hash_combine(seed, eps);
    }
    bool operator==(const RmsRopeKey& rhs) const {
        return dim == rhs.dim && rotary_dim == rhs.rotary_dim && eps == rhs.eps;
    }
};

// The RMSNorm of an f32 row followed by the RoPE of its first `rotary_dim` channels, on the JIT kernels the RMSNorm
// and the RoPE nodes use (AVX-512 or AVX2):
//   dst = src / sqrt(mean(src^2) + eps) * weight         step 1, [dim]
//   dst[0, rotary_dim) rotated by cos / sin [rotary_dim]  step 2, rotate-half layout
// `dst` may be `src`.
class JitRmsRope {
public:
    // The kernels of `key` from the params cache (built on first use, shared by the layers), nullptr if the CPU has
    // neither AVX-512 nor AVX2.
    static std::shared_ptr<const JitRmsRope> get(const MultiCachePtr& cache, const RmsRopeKey& key);

    JitRmsRope(std::shared_ptr<kernel::JitKernelBase> rms, std::shared_ptr<kernel::JitKernelBase> rotary);

    void operator()(const float* src, float* dst, const float* weight, const float* cos, const float* sin) const;

private:
    std::shared_ptr<kernel::JitKernelBase> m_rms;
    std::shared_ptr<kernel::JitKernelBase> m_rotary;  // null when nothing is rotated
};

// What the JIT scoring kernel is specialized for; also the key in the params cache.
struct QsaScoreKey {
    ov::element::Type row_prc;  // precision of the summary rows
    size_t dim = 0;             // channels of a row
    size_t heads = 0;           // query heads of an index key head

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, row_prc.hash());
        seed = dnnl::impl::hash_combine(seed, dim);
        return dnnl::impl::hash_combine(seed, heads);
    }
    bool operator==(const QsaScoreKey& rhs) const {
        return row_prc == rhs.row_prc && dim == rhs.dim && heads == rhs.heads;
    }
};

// The scores of summary rows against the query heads of an index key head, on a JIT kernel (AVX-512 or AVX2):
//   scores[i] += sum over the heads h of max(0, q[h] . rows[i])        i < n, q [heads, dim] f32, rows [n, dim]
class JitQsaScore {
public:
    // Whether this CPU has the kernel for such rows: the channels have to fill whole vectors.
    static bool supports(const ov::element::Type& row_prc, size_t dim);

    // The kernel of `key` from the params cache (built on first use, shared by the layers), nullptr if the CPU does
    // not support it.
    static std::shared_ptr<const JitQsaScore> get(const MultiCachePtr& cache, const QsaScoreKey& key);

    explicit JitQsaScore(std::shared_ptr<kernel::JitKernelBase> kernel);

    // `stage` is scratch of 4 * dim floats.
    void operator()(const float* q, const void* rows, size_t n, float* scores, float* stage) const;

private:
    std::shared_ptr<kernel::JitKernelBase> m_kernel;
};

}  // namespace ov::intel_cpu::node::qsa
