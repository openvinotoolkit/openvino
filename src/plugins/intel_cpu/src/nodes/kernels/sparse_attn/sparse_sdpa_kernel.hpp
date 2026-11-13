// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu {

/// @brief What is fixed for the lifetime of a SparseSDPA node. A cache is compressed when its spec holds u8 / u4 (see
/// CacheSpec: per-token groups of `group_size` elements, or per-channel groups of tokens with `by_channel`); any other
/// precision is a plain cache read as is.
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
    CacheSpec key_spec;
    CacheSpec value_spec;
    /// The AMX prefill kernels; not enabled when the node is not eligible. They are owned by the node and must outlive
    /// the executor.
    TileBrgemm brgemm;
};

/// @brief The tensors of one call. Query, key, value and output share one floating-point type (f32 / f16 / bf16),
/// except the key / value of a compressed cache, which are the u8 / u4 rows with {scale, zero point} in `*_scale_zp`.
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

/// @brief Block-sparse group-query attention of the SparseSDPA node over a dense BHLS key / value cache. Built once by
/// the factory below; the node asks for the scratch size of a call, allocates it and passes it to execute.
class SparseSdpaExecutor {
public:
    virtual ~SparseSdpaExecutor() = default;
    [[nodiscard]] virtual size_t scratch_bytes(const SparseSdpaTensors& tensors,
                                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
    virtual void execute(const SparseSdpaTensors& tensors,
                         uint8_t* scratch,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
};

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

// Cross-compiled factory (AVX-512, AVX2 or plain C++ by the CPU): the SIMD decode, the AMX prefill and the row
// conversion live in sparse_sdpa_kernel.cpp so the per-ISA clones vectorize them.
std::shared_ptr<ov::Extensions::Cpu::SparseSdpaExecutor> make_sparse_sdpa_executor(
    const ov::Extensions::Cpu::SparseSdpaConfig& config);

// Dequantizes the rows [0, L) of the compressed cache (b, h) into dense f32 rows, `dst` is [L, D].
void sparse_kv_dequantize_rows(const ov::intel_cpu::PlainTensor& cache,
                               const ov::Extensions::Cpu::CacheSpec& spec,
                               const ov::intel_cpu::PlainTensor& scale_zp,
                               size_t b,
                               size_t h,
                               size_t L,
                               size_t D,
                               float* dst);

}  // namespace ov::Extensions::Cpu::XARCH
