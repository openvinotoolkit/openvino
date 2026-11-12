// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ov::intel_gpu::kv_chunk_table {

// Layout of the i64 tensor bound to a PagedAttention key_cache/value_cache port in chunk-table mode.
// Granule = 2^granule_shift blocks; every chunk holds a whole number of granules and block ids (GBID) are dense
// across chunks in allocation order.
//   word 0                    : granule_shift (bits 0..7) | num_chunks (bits 8..31) | generation (bits 32..63)
//   word 1                    : num_granules
//   words [2, 2 + num_chunks) : chunk base pointers (this layer, K or V)
//   then, num_granules u32    : (chunk_id << 20) | granule_in_chunk, two entries per word, low half first
// The OpenCL kernels decode this layout in pa_resolve_chunk_block().
constexpr uint32_t kGranuleInChunkBits = 20;
constexpr uint32_t kMaxChunks = 1u << 12;
constexpr uint32_t kMaxGranulesPerChunk = 1u << kGranuleInChunkBits;
constexpr size_t kHeaderWords = 2;

struct Header {
    uint32_t granule_shift = 0;
    uint32_t num_chunks = 0;
    uint32_t generation = 0;
    uint64_t num_granules = 0;
};

inline Header read_header(const int64_t* words) {
    const auto word0 = static_cast<uint64_t>(words[0]);
    Header header;
    header.granule_shift = static_cast<uint32_t>(word0 & 0xFFu);
    header.num_chunks = static_cast<uint32_t>((word0 >> 8) & 0xFFFFFFu);
    header.generation = static_cast<uint32_t>(word0 >> 32);
    header.num_granules = static_cast<uint64_t>(words[1]);
    return header;
}

// Builds a table from chunk base pointers and per-chunk block counts (each a multiple of 2^granule_shift).
inline std::vector<int64_t> build(uint32_t granule_shift,
                                  uint32_t generation,
                                  const std::vector<int64_t>& chunk_base_ptrs,
                                  const std::vector<size_t>& chunk_blocks) {
    if (chunk_base_ptrs.size() != chunk_blocks.size() || chunk_base_ptrs.size() > kMaxChunks) {
        throw std::invalid_argument("kv_chunk_table: invalid number of chunks");
    }
    const size_t granule_blocks = size_t{1} << granule_shift;
    std::vector<uint32_t> directory;
    for (size_t chunk = 0; chunk < chunk_blocks.size(); ++chunk) {
        if (chunk_blocks[chunk] == 0 || chunk_blocks[chunk] % granule_blocks != 0 ||
            chunk_blocks[chunk] / granule_blocks > kMaxGranulesPerChunk) {
            throw std::invalid_argument("kv_chunk_table: chunk size must be a non-zero multiple of the granule size");
        }
        for (size_t granule = 0; granule < chunk_blocks[chunk] / granule_blocks; ++granule) {
            directory.push_back(static_cast<uint32_t>((chunk << kGranuleInChunkBits) | granule));
        }
    }
    const size_t num_chunks = chunk_base_ptrs.size();
    const size_t directory_words = (directory.size() + 1) / 2;
    std::vector<int64_t> table(kHeaderWords + num_chunks + directory_words, 0);
    table[0] = static_cast<int64_t>(static_cast<uint64_t>(granule_shift) | (static_cast<uint64_t>(num_chunks) << 8) |
                                    (static_cast<uint64_t>(generation) << 32));
    table[1] = static_cast<int64_t>(directory.size());
    std::copy(chunk_base_ptrs.begin(), chunk_base_ptrs.end(), table.begin() + kHeaderWords);
    std::memcpy(table.data() + kHeaderWords + num_chunks, directory.data(), directory.size() * sizeof(uint32_t));
    return table;
}

}  // namespace ov::intel_gpu::kv_chunk_table
