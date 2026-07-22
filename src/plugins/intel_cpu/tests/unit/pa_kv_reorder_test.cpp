// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/cache_reorder.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"

using namespace ov::intel_cpu;
using namespace ov::Extensions::Cpu::XARCH;

namespace {

// Common helpers shared by all PaKVReorder kernel tests.
class PaKVReorderTestBase : public ::testing::Test {
protected:
    static constexpr size_t kBlockSize = 32;
    static constexpr size_t kHeadSize = 64;
    static constexpr size_t kNumHeads = 2;
    // Typical uint8 quantization: data range [0, 255] mapped via (q - 128) * 0.1f.
    static constexpr float kDefaultScale = 0.1f;
    static constexpr float kDefaultZp = 128.0f;

    CpuParallelPtr make_cpu_parallel() {
        return std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    }

    // Build a 1D PlainTensor over `storage` (caller owns the memory).
    template <typename T>
    static PlainTensor make_index_tensor(std::vector<T>& storage) {
        PlainTensor t;
        t.template resize<T>({storage.size()}, storage.data());
        return t;
    }
};

// Fixture for f32 / non-quantized KV cache tests.
class PaKVReorderFloatCacheTest : public PaKVReorderTestBase {
protected:
    // Fill cache buffers so each (block, token) is uniquely identifiable. `value_offset` lets the
    // value tensor differ from the key tensor without changing the per-token signature.
    void init_float_cache(size_t num_blocks,
                          std::vector<float>& key_data,
                          std::vector<float>& value_data,
                          float value_offset = 50000.0f) {
        const size_t total = num_blocks * kNumHeads * kBlockSize * kHeadSize;
        key_data.resize(total);
        value_data.resize(total);

        for (size_t b = 0; b < num_blocks; b++) {
            for (size_t h = 0; h < kNumHeads; h++) {
                for (size_t t = 0; t < kBlockSize; t++) {
                    for (size_t d = 0; d < kHeadSize; d++) {
                        const size_t idx =
                            b * kNumHeads * kBlockSize * kHeadSize + h * kBlockSize * kHeadSize + t * kHeadSize + d;
                        const float token_id = static_cast<float>(b * 1000 + t);
                        key_data[idx] = token_id;
                        value_data[idx] = token_id + value_offset;
                    }
                }
            }
        }
    }

    static size_t cache_offset(size_t block, size_t head, size_t token) {
        return block * kNumHeads * kBlockSize * kHeadSize + head * kBlockSize * kHeadSize + token * kHeadSize;
    }

    static void expect_token_payload_equals(const std::vector<float>& dst_data,
                                            const std::vector<float>& src_data_orig,
                                            size_t dst_block,
                                            size_t dst_token,
                                            size_t src_block,
                                            size_t src_token) {
        for (size_t h = 0; h < kNumHeads; h++) {
            const size_t src_off = cache_offset(src_block, h, src_token);
            const size_t dst_off = cache_offset(dst_block, h, dst_token);
            for (size_t d = 0; d < kHeadSize; d++) {
                EXPECT_FLOAT_EQ(dst_data[dst_off + d], src_data_orig[src_off + d])
                    << "mismatch at head=" << h << ", dim=" << d;
            }
        }
    }
};

// Fixture for u8 quantized KV cache tests.
class PaKVReorderQuantizedCacheTest : public PaKVReorderTestBase {
protected:
    static constexpr size_t kByChannelParamsBytes = 2 * sizeof(float) * kHeadSize;  // scale[H] + zp[H]
    static constexpr size_t kByChannelDataBytes = kBlockSize * kHeadSize;
    static constexpr size_t kByChannelBlockHeadBytes = kByChannelParamsBytes + kByChannelDataBytes;

    static constexpr size_t kByTokenParamsBytes = 2 * sizeof(float);  // scale + zp per token
    static constexpr size_t kByTokenStrideBytes = kByTokenParamsBytes + kHeadSize;
    static constexpr size_t kByTokenBlockHeadBytes = kBlockSize * kByTokenStrideBytes;

    // Fill one (block, head) of a by-channel quantized cache with kDefaultScale / kDefaultZp and a
    // recognizable quant payload.
    static void init_by_channel_block_head(uint8_t* base, size_t block_id, size_t pattern_offset) {
        auto* scales = reinterpret_cast<float*>(base);
        auto* zps = reinterpret_cast<float*>(base + sizeof(float) * kHeadSize);
        for (size_t d = 0; d < kHeadSize; d++) {
            scales[d] = kDefaultScale;
            zps[d] = kDefaultZp;
        }
        uint8_t* quant = base + kByChannelParamsBytes;
        for (size_t t = 0; t < kBlockSize; t++) {
            for (size_t d = 0; d < kHeadSize; d++) {
                quant[t * kHeadSize + d] = static_cast<uint8_t>((block_id * 10 + t + pattern_offset) % 256);
            }
        }
    }

    // Fill one (block, head) of a by-token (interleaved) quantized cache. Each token row contains
    // [scale, zp, data[head_size]].
    static void init_by_token_block_head(uint8_t* base, size_t block_id, size_t pattern_offset) {
        for (size_t t = 0; t < kBlockSize; t++) {
            uint8_t* token_ptr = base + t * kByTokenStrideBytes;
            auto* scale_ptr = reinterpret_cast<float*>(token_ptr);
            auto* zp_ptr = reinterpret_cast<float*>(token_ptr + sizeof(float));
            *scale_ptr = kDefaultScale;
            *zp_ptr = kDefaultZp;

            uint8_t* data_ptr = token_ptr + kByTokenParamsBytes;
            for (size_t d = 0; d < kHeadSize; d++) {
                data_ptr[d] = static_cast<uint8_t>((block_id * 10 + t + pattern_offset) % 256);
            }
        }
    }

    // PlainTensor describing a by-channel cache buffer. dim2 = block_size + 2*sizeof(float) packs
    // params and data along the same axis; explicit strides preserve the actual byte layout.
    static PlainTensor wrap_by_channel(uint8_t* buf, size_t num_blocks) {
        const size_t dim2_with_params = kBlockSize + 2 * sizeof(float);  // 32 + 8 = 40
        const size_t strides[4] = {
            kNumHeads * kByChannelBlockHeadBytes,  // bytes per block
            kByChannelBlockHeadBytes,              // bytes per head
            kHeadSize,                             // bytes per token-or-params row
            1                                      // bytes per element
        };
        PlainTensor t;
        t.resize<uint8_t>({num_blocks, kNumHeads, dim2_with_params, kHeadSize}, buf, strides);
        return t;
    }

    // PlainTensor describing a by-token (interleaved) cache buffer. The hidden axis carries
    // head_size + scale + zp; stride[2] is the byte distance between successive tokens.
    static PlainTensor wrap_by_token(uint8_t* buf, size_t num_blocks) {
        const size_t dim3_with_params = kHeadSize + 2 * sizeof(float);  // 64 + 8 = 72
        const size_t strides[4] = {
            kNumHeads * kByTokenBlockHeadBytes,  // bytes per block
            kByTokenBlockHeadBytes,              // bytes per head
            kByTokenStrideBytes,                 // bytes per token (interleaved)
            1                                    // bytes per element
        };
        PlainTensor t;
        t.resize<uint8_t>({num_blocks, kNumHeads, kBlockSize, dim3_with_params}, buf, strides);
        return t;
    }

    static float dequantize(uint8_t q, float zp, float scale) {
        return (static_cast<float>(q) - zp) * scale;
    }

    // Compare dequantized rows token-by-token. After requantization the raw u8 values can shift,
    // so verify in float space using each side's own scale/zp.
    static void expect_by_channel_token_dequant_close(const std::vector<uint8_t>& cache_after,
                                                      const std::vector<uint8_t>& cache_before,
                                                      size_t block,
                                                      size_t dst_token,
                                                      size_t src_token,
                                                      float tolerance) {
        for (size_t h = 0; h < kNumHeads; h++) {
            const size_t base_offset = (block * kNumHeads + h) * kByChannelBlockHeadBytes;

            const auto* scales_after = reinterpret_cast<const float*>(cache_after.data() + base_offset);
            const auto* zps_after =
                reinterpret_cast<const float*>(cache_after.data() + base_offset + sizeof(float) * kHeadSize);
            const auto* scales_before = reinterpret_cast<const float*>(cache_before.data() + base_offset);
            const auto* zps_before =
                reinterpret_cast<const float*>(cache_before.data() + base_offset + sizeof(float) * kHeadSize);

            const uint8_t* data_after = cache_after.data() + base_offset + kByChannelParamsBytes;
            const uint8_t* data_before = cache_before.data() + base_offset + kByChannelParamsBytes;

            for (size_t d = 0; d < kHeadSize; d++) {
                const float dst_dequant =
                    dequantize(data_after[dst_token * kHeadSize + d], zps_after[d], scales_after[d]);
                const float src_dequant =
                    dequantize(data_before[src_token * kHeadSize + d], zps_before[d], scales_before[d]);
                EXPECT_NEAR(dst_dequant, src_dequant, tolerance) << "head=" << h << " dim=" << d;
            }
        }
    }

    // Verify a by-token (interleaved) update: scale, zp, and quantized payload should be byte-equal
    // because by-token reorder is a direct memcpy of the whole token record.
    static void expect_by_token_record_equal(const std::vector<uint8_t>& cache_after,
                                             const std::vector<uint8_t>& cache_before,
                                             size_t block,
                                             size_t dst_token,
                                             size_t src_token) {
        for (size_t h = 0; h < kNumHeads; h++) {
            const size_t base_offset = (block * kNumHeads + h) * kByTokenBlockHeadBytes;
            const uint8_t* dst_token_ptr = cache_after.data() + base_offset + dst_token * kByTokenStrideBytes;
            const uint8_t* src_token_ptr = cache_before.data() + base_offset + src_token * kByTokenStrideBytes;

            const auto* dst_scale = reinterpret_cast<const float*>(dst_token_ptr);
            const auto* dst_zp = reinterpret_cast<const float*>(dst_token_ptr + sizeof(float));
            const auto* src_scale = reinterpret_cast<const float*>(src_token_ptr);
            const auto* src_zp = reinterpret_cast<const float*>(src_token_ptr + sizeof(float));
            EXPECT_FLOAT_EQ(*dst_scale, *src_scale) << "head=" << h;
            EXPECT_FLOAT_EQ(*dst_zp, *src_zp) << "head=" << h;

            const uint8_t* dst_data = dst_token_ptr + kByTokenParamsBytes;
            const uint8_t* src_data = src_token_ptr + kByTokenParamsBytes;
            for (size_t d = 0; d < kHeadSize; d++) {
                EXPECT_EQ(dst_data[d], src_data[d]) << "head=" << h << " dim=" << d;
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Float / non-quantized cache tests
// ----------------------------------------------------------------------------

TEST_F(PaKVReorderFloatCacheTest, SameBlock) {
    constexpr size_t num_blocks = 2;

    std::vector<float> key_data;
    std::vector<float> value_data;
    init_float_cache(num_blocks, key_data, value_data, 100000.0f);
    const auto key_data_copy = key_data;
    const auto value_data_copy = value_data;

    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, kNumHeads, kBlockSize, kHeadSize}, key_data.data());
    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, kNumHeads, kBlockSize, kHeadSize}, value_data.data());

    std::vector<int32_t> block_indices_data = {0, 1};
    std::vector<int32_t> block_indices_begins_data = {0, 2};
    std::vector<int32_t> block_update_indices_data = {10, 5};  // src=10, dst=5 (backward)
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     /*key_by_channel=*/false,
                     /*value_by_channel=*/false,
                     make_cpu_parallel());

    // Token 5 should now hold token 10's original payload (within block 0).
    expect_token_payload_equals(key_data, key_data_copy, 0, 5, 0, 10);
    expect_token_payload_equals(value_data, value_data_copy, 0, 5, 0, 10);
}

TEST_F(PaKVReorderFloatCacheTest, CrossBlock) {
    constexpr size_t num_blocks = 4;

    std::vector<float> key_data;
    std::vector<float> value_data;
    init_float_cache(num_blocks, key_data, value_data);
    const auto key_data_copy = key_data;
    const auto value_data_copy = value_data;

    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, kNumHeads, kBlockSize, kHeadSize}, key_data.data());
    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, kNumHeads, kBlockSize, kHeadSize}, value_data.data());

    // Sequence uses physical blocks {0, 1, 2}; copy logical token 5 (in block 0) to
    // logical token 42 = 32 + 10 (in block 1).
    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {5, 42};
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     /*key_by_channel=*/false,
                     /*value_by_channel=*/false,
                     make_cpu_parallel());

    expect_token_payload_equals(key_data, key_data_copy, 1, 10, 0, 5);
    expect_token_payload_equals(value_data, value_data_copy, 1, 10, 0, 5);
}

// ----------------------------------------------------------------------------
// Quantized cache tests
// ----------------------------------------------------------------------------

TEST_F(PaKVReorderQuantizedCacheTest, BothByChannel) {
    constexpr size_t num_blocks = 4;

    std::vector<uint8_t> key_cache_data(num_blocks * kNumHeads * kByChannelBlockHeadBytes);
    std::vector<uint8_t> value_cache_data(num_blocks * kNumHeads * kByChannelBlockHeadBytes);

    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < kNumHeads; h++) {
            const size_t off = (b * kNumHeads + h) * kByChannelBlockHeadBytes;
            init_by_channel_block_head(key_cache_data.data() + off, b, /*pattern_offset=*/0);
            init_by_channel_block_head(value_cache_data.data() + off, b, /*pattern_offset=*/50);
        }
    }

    const auto key_cache_copy = key_cache_data;
    const auto value_cache_copy = value_cache_data;

    PlainTensor key_cache = wrap_by_channel(key_cache_data.data(), num_blocks);
    PlainTensor value_cache = wrap_by_channel(value_cache_data.data(), num_blocks);

    // Same-block backward copy: token 10 → token 5 in block 0.
    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {10, 5};
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     /*key_by_channel=*/true,
                     /*value_by_channel=*/true,
                     make_cpu_parallel());

    // For by-channel the requantize step may reshape scale/zp, so compare in dequantized space.
    expect_by_channel_token_dequant_close(key_cache_data, key_cache_copy, 0, 5, 10, 0.2f);
    expect_by_channel_token_dequant_close(value_cache_data, value_cache_copy, 0, 5, 10, 0.2f);
}

TEST_F(PaKVReorderQuantizedCacheTest, KeyByChannelValueByToken) {
    constexpr size_t num_blocks = 4;

    std::vector<uint8_t> key_cache_data(num_blocks * kNumHeads * kByChannelBlockHeadBytes);
    std::vector<uint8_t> value_cache_data(num_blocks * kNumHeads * kByTokenBlockHeadBytes);

    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < kNumHeads; h++) {
            const size_t key_off = (b * kNumHeads + h) * kByChannelBlockHeadBytes;
            init_by_channel_block_head(key_cache_data.data() + key_off, b, /*pattern_offset=*/0);

            const size_t val_off = (b * kNumHeads + h) * kByTokenBlockHeadBytes;
            init_by_token_block_head(value_cache_data.data() + val_off, b, /*pattern_offset=*/50);
        }
    }

    const auto key_cache_copy = key_cache_data;
    const auto value_cache_copy = value_cache_data;

    PlainTensor key_cache = wrap_by_channel(key_cache_data.data(), num_blocks);
    PlainTensor value_cache = wrap_by_token(value_cache_data.data(), num_blocks);

    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {10, 5};
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     /*key_by_channel=*/true,
                     /*value_by_channel=*/false,
                     make_cpu_parallel());

    // Key path: by-channel, so compare dequantized values.
    expect_by_channel_token_dequant_close(key_cache_data, key_cache_copy, 0, 5, 10, 0.2f);
    // Value path: by-token, so the whole token record (scale, zp, quant) is byte-copied.
    expect_by_token_record_equal(value_cache_data, value_cache_copy, 0, 5, 10);
}

// ----------------------------------------------------------------------------
// Error-path tests (no need for cache fixtures — only validation is exercised).
// ----------------------------------------------------------------------------

TEST_F(PaKVReorderTestBase, Int8CacheIsRejected) {
    constexpr size_t num_blocks = 1;
    constexpr size_t num_heads = 1;
    constexpr size_t block_size = 4;
    constexpr size_t head_size = 8;

    std::vector<int8_t> key_data(num_blocks * num_heads * block_size * head_size, 1);
    std::vector<int8_t> value_data(num_blocks * num_heads * block_size * head_size, 2);

    PlainTensor key_cache;
    key_cache.resize<int8_t>({num_blocks, num_heads, block_size, head_size}, key_data.data());
    PlainTensor value_cache;
    value_cache.resize<int8_t>({num_blocks, num_heads, block_size, head_size}, value_data.data());

    std::vector<int32_t> block_indices_data = {0};
    std::vector<int32_t> block_indices_begins_data = {0, 1};
    std::vector<int32_t> block_update_indices_data = {0, 1};
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    EXPECT_THROW(reorder_kv_cache(key_cache,
                                  value_cache,
                                  block_indices,
                                  block_indices_begins,
                                  block_update_indices,
                                  block_update_indices_begins,
                                  false,
                                  false,
                                  make_cpu_parallel()),
                 ov::Exception);
}

TEST_F(PaKVReorderTestBase, InvalidBlockUpdateBeginsAreRejected) {
    constexpr size_t num_blocks = 1;
    constexpr size_t num_heads = 1;
    constexpr size_t block_size = 4;
    constexpr size_t head_size = 8;

    std::vector<float> key_data(num_blocks * num_heads * block_size * head_size, 1.f);
    std::vector<float> value_data(num_blocks * num_heads * block_size * head_size, 2.f);

    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, key_data.data());
    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, value_data.data());

    std::vector<int32_t> block_indices_data = {0};
    std::vector<int32_t> block_indices_begins_data = {0, 1};
    std::vector<int32_t> block_update_indices_data = {0, 1};
    // Non-monotonic offsets must be rejected.
    std::vector<int32_t> block_update_indices_begins_data = {1, 0};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    EXPECT_THROW(reorder_kv_cache(key_cache,
                                  value_cache,
                                  block_indices,
                                  block_indices_begins,
                                  block_update_indices,
                                  block_update_indices_begins,
                                  false,
                                  false,
                                  make_cpu_parallel()),
                 ov::Exception);
}

TEST_F(PaKVReorderTestBase, InvalidBlockIndicesBeginsAreRejected) {
    constexpr size_t num_blocks = 1;
    constexpr size_t num_heads = 1;
    constexpr size_t block_size = 4;
    constexpr size_t head_size = 8;

    std::vector<float> key_data(num_blocks * num_heads * block_size * head_size, 1.f);
    std::vector<float> value_data(num_blocks * num_heads * block_size * head_size, 2.f);

    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, key_data.data());
    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, value_data.data());

    std::vector<int32_t> block_indices_data = {0};
    // begins[1] = 2 > block_indices.size() = 1 → out of range.
    std::vector<int32_t> block_indices_begins_data = {0, 2};
    std::vector<int32_t> block_update_indices_data = {0, 1};
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    auto block_indices = make_index_tensor(block_indices_data);
    auto block_indices_begins = make_index_tensor(block_indices_begins_data);
    auto block_update_indices = make_index_tensor(block_update_indices_data);
    auto block_update_indices_begins = make_index_tensor(block_update_indices_begins_data);

    EXPECT_THROW(reorder_kv_cache(key_cache,
                                  value_cache,
                                  block_indices,
                                  block_indices_begins,
                                  block_update_indices,
                                  block_update_indices_begins,
                                  false,
                                  false,
                                  make_cpu_parallel()),
                 ov::Exception);
}

}  // namespace
