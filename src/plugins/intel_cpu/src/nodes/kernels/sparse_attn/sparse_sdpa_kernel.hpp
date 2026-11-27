// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cache/multi_cache.h"
#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu {

struct SparseSdpaConfig {
    size_t num_heads = 0;            ///< query heads
    size_t num_kv_heads = 0;         ///< key / value heads
    size_t num_selection_heads = 1;  ///< heads of the block selection: 1, num_kv_heads or num_heads
    size_t k_head_size = 0;          ///< E of the query and the keys
    size_t v_head_size = 0;          ///< Ev of the values and the output
    ov::element::Type data_precision = ov::element::f32;  ///< query, output and plain key / value type
    size_t key_block_size = 0;
    bool causal = true;
    float scale = 1.0F;
    ov::intel_cpu::CacheSpec key_spec;
    ov::intel_cpu::CacheSpec value_spec;
    /// The AMX prefill kernels; empty when the executor is not eligible. make_*_executor() fills it (the node leaves it
    /// empty). The config owns them, so an executor that copies the config keeps them alive.
    TileBrgemmKernels brgemm;
};

struct SparseSdpaTensors {
    const ov::intel_cpu::PlainTensor& query;        ///< [N, H, L, E]
    const ov::intel_cpu::PlainTensor& key;          ///< [N, Hkv, S, E]
    const ov::intel_cpu::PlainTensor& value;        ///< [N, Hkv, S, Ev]
    const ov::intel_cpu::PlainTensor& sel_indices;  ///< [N, Hs, L, KB]
    const ov::intel_cpu::PlainTensor& sel_count;    ///< [N, Hs, L]
    ov::intel_cpu::PlainTensor& out;                ///< [N, H, L, Ev]
    const int32_t* key_length = nullptr;            ///< optional, 1 or N entries
    size_t key_length_batch_stride = 0;             ///< 0: one key_length for all the batches, 1: one per batch
    const ov::intel_cpu::PlainTensor* key_scale_zp = nullptr;    ///< compressed key cache only
    const ov::intel_cpu::PlainTensor* value_scale_zp = nullptr;  ///< compressed value cache only
};

class SparseSdpaExecutor {
public:
    virtual ~SparseSdpaExecutor() = default;
    /// The implementation type the node reports: the matrix extension of the prefill kernels when it has them, else
    /// the SIMD stages of the variant.
    [[nodiscard]] virtual ov::intel_cpu::impl_desc_type impl_type() const = 0;
    [[nodiscard]] virtual size_t scratch_bytes(const SparseSdpaTensors& tensors,
                                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
    virtual void execute(const SparseSdpaTensors& tensors,
                         const ov::intel_cpu::MemoryPtr& scratch,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
};

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

std::shared_ptr<ov::Extensions::Cpu::SparseSdpaExecutor> make_sparse_sdpa_executor(
    const ov::Extensions::Cpu::SparseSdpaConfig& config,
    const ov::intel_cpu::MultiCachePtr& cache);

// Dequantizes the rows [0, L) of the compressed cache (b, h) into dense f32 rows, `dst` is [L, D].
void sparse_kv_dequantize_rows(const ov::intel_cpu::PlainTensor& cache,
                               const ov::intel_cpu::CacheSpec& spec,
                               const ov::intel_cpu::PlainTensor& scale_zp,
                               size_t b,
                               size_t h,
                               size_t L,
                               size_t D,
                               float* dst);

}  // namespace ov::Extensions::Cpu::XARCH
