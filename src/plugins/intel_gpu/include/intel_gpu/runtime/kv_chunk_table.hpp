// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ov::intel_gpu::kv_chunk_table {

// Layout of the global KV chunk table (PagedAttentionExtension input 28, an i32 tensor holding 2 * total_slots
// elements; every u64 slot is stored as two i32 elements, low half first). One table is shared by all layers:
//
//   slots [0, 16)                    header (see HeaderSlot)
//   slots [16, 16 + ceil(D_max/2))   directory: D_max u32 entries, two per slot, low half first:
//                                    dir[k] = (chunk_id << 20) | granule_in_chunk
//   then, per layer l, 2 * C_max slots: K chunk pointers [0, C_max) followed by V chunk pointers [0, C_max)
//
// A granule is 2^granule_shift consecutive blocks; every chunk holds a whole number of granules and global block ids
// (GBID) are dense across chunks in allocation order. The OpenCL kernels resolve a block as
//   entry       = dir[gbid >> g]
//   local_block = ((entry & 0xFFFFF) << g) | (gbid & ((1 << g) - 1))
//   address     = slot[bases_offset + layer * 2 * C_max + (is_value ? C_max : 0) + (entry >> 20)] + local_block * block_bytes
// (see pa_resolve_chunk_block() in the .cl files).
constexpr uint64_t kMagic = 0x50414348554E4B33ull;  // "PACHUNK3"
constexpr uint32_t kGranuleInChunkBits = 20;
constexpr uint32_t kMaxChunkId = (1u << 12) - 1;
constexpr uint32_t kMaxGranulesPerChunk = 1u << kGranuleInChunkBits;
constexpr size_t kHeaderSlots = 16;

enum HeaderSlot : size_t {
    MAGIC = 0,
    NUM_LAYERS = 1,
    MAX_CHUNKS = 2,
    LIVE_CHUNKS = 3,
    GRANULE_SHIFT = 4,
    LIVE_GRANULES = 5,
    MAX_GRANULES = 6,
    GENERATION = 8,
    FLAGS = 9,
};

struct Header {
    uint32_t num_layers = 0;
    uint32_t max_chunks = 0;
    uint32_t live_chunks = 0;
    uint32_t granule_shift = 0;
    uint64_t live_granules = 0;
    uint32_t max_granules = 0;
    uint64_t generation = 0;
};

inline size_t directory_slots(size_t max_granules) {
    return (max_granules + 1) / 2;
}

inline size_t bases_offset(size_t max_granules) {
    return kHeaderSlots + directory_slots(max_granules);
}

inline size_t total_slots(size_t num_layers, size_t max_chunks, size_t max_granules) {
    return bases_offset(max_granules) + num_layers * 2 * max_chunks;
}

// First slot of the K chunk pointers of `layer`; V pointers start max_chunks slots later.
inline size_t layer_base_slot(const Header& header, size_t layer) {
    return bases_offset(header.max_granules) + layer * 2 * header.max_chunks;
}

inline bool is_valid(const uint64_t* slots) {
    return slots[MAGIC] == kMagic;
}

inline Header read_header(const uint64_t* slots) {
    Header header;
    header.num_layers = static_cast<uint32_t>(slots[NUM_LAYERS]);
    header.max_chunks = static_cast<uint32_t>(slots[MAX_CHUNKS]);
    header.live_chunks = static_cast<uint32_t>(slots[LIVE_CHUNKS]);
    header.granule_shift = static_cast<uint32_t>(slots[GRANULE_SHIFT]);
    header.live_granules = slots[LIVE_GRANULES];
    header.max_granules = static_cast<uint32_t>(slots[MAX_GRANULES]);
    header.generation = slots[GENERATION];
    return header;
}

// Test/utility builder. `chunk_blocks[c]` is the block count of chunk c (a non-zero multiple of 2^granule_shift);
// `key_ptrs[l][c]` / `value_ptrs[l][c]` are the base pointers of chunk c for layer l.
inline std::vector<uint64_t> build(uint32_t granule_shift,
                                   uint64_t generation,
                                   uint32_t max_chunks,
                                   uint32_t max_granules,
                                   const std::vector<size_t>& chunk_blocks,
                                   const std::vector<std::vector<uint64_t>>& key_ptrs,
                                   const std::vector<std::vector<uint64_t>>& value_ptrs) {
    if (key_ptrs.size() != value_ptrs.size() || chunk_blocks.size() > max_chunks || chunk_blocks.size() > kMaxChunkId + 1) {
        throw std::invalid_argument("kv_chunk_table: invalid number of layers/chunks");
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
    if (directory.size() > max_granules) {
        throw std::invalid_argument("kv_chunk_table: granule directory capacity exceeded");
    }
    const size_t num_layers = key_ptrs.size();
    std::vector<uint64_t> slots(total_slots(num_layers, max_chunks, max_granules), 0);
    slots[MAGIC] = kMagic;
    slots[NUM_LAYERS] = num_layers;
    slots[MAX_CHUNKS] = max_chunks;
    slots[LIVE_CHUNKS] = chunk_blocks.size();
    slots[GRANULE_SHIFT] = granule_shift;
    slots[LIVE_GRANULES] = directory.size();
    slots[MAX_GRANULES] = max_granules;
    slots[GENERATION] = generation;
    std::memcpy(slots.data() + kHeaderSlots, directory.data(), directory.size() * sizeof(uint32_t));
    const Header header = read_header(slots.data());
    for (size_t layer = 0; layer < num_layers; ++layer) {
        if (key_ptrs[layer].size() != chunk_blocks.size() || value_ptrs[layer].size() != chunk_blocks.size()) {
            throw std::invalid_argument("kv_chunk_table: per-layer pointer count must match the chunk count");
        }
        const size_t base = layer_base_slot(header, layer);
        std::copy(key_ptrs[layer].begin(), key_ptrs[layer].end(), slots.begin() + base);
        std::copy(value_ptrs[layer].begin(), value_ptrs[layer].end(), slots.begin() + base + max_chunks);
    }
    return slots;
}

}  // namespace ov::intel_gpu::kv_chunk_table
