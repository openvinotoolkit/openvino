// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/cache_reorder.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"
using namespace ov::intel_cpu;
using namespace ov::Extensions::Cpu::XARCH;

// Test actual reorder_kv_cache kernel with float data
TEST(PaKVReorderKernelTest, FloatCacheSameBlock) {
    using namespace ov::intel_cpu;
    using namespace ov::Extensions::Cpu::XARCH;

    constexpr size_t num_blocks = 2;
    constexpr size_t num_heads = 2;
    constexpr size_t block_size = 32;
    constexpr size_t head_size = 64;

    // Create key/value cache data
    std::vector<float> key_data(num_blocks * num_heads * block_size * head_size);
    std::vector<float> value_data(num_blocks * num_heads * block_size * head_size);

    // Initialize with unique values
    for (size_t i = 0; i < key_data.size(); i++) {
        key_data[i] = static_cast<float>(i);
        value_data[i] = static_cast<float>(i + 100000);
    }

    // Save original data for verification
    std::vector<float> key_data_copy = key_data;
    std::vector<float> value_data_copy = value_data;

    // Wrap in PlainTensor
    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, key_data.data());

    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, value_data.data());

    // Prepare indices: single sequence, copy token 5->10 in block 0
    std::vector<int32_t> block_indices_data = {0, 1};
    std::vector<int32_t> block_indices_begins_data = {0, 2};
    std::vector<int32_t> block_update_indices_data = {10, 5};  // src=10, dst=5 (backward)  // (src, dst)
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    PlainTensor block_indices;
    block_indices.resize<int32_t>({2}, block_indices_data.data());

    PlainTensor block_indices_begins;
    block_indices_begins.resize<int32_t>({2}, block_indices_begins_data.data());

    PlainTensor block_update_indices;
    block_update_indices.resize<int32_t>({2}, block_update_indices_data.data());

    PlainTensor block_update_indices_begins;
    block_update_indices_begins.resize<int32_t>({2}, block_update_indices_begins_data.data());

    // Execute kernel
    auto cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     false,  // key_by_channel
                     false,  // value_by_channel
                     cpu_parallel);

    // Verify: token 5 (dst) should have data from token 10 (src)
    for (size_t h = 0; h < num_heads; h++) {
        size_t src_offset = 0 * num_heads * block_size * head_size + h * block_size * head_size + 10 * head_size;
        size_t dst_offset = 0 * num_heads * block_size * head_size + h * block_size * head_size + 5 * head_size;

        for (size_t d = 0; d < head_size; d++) {
            EXPECT_FLOAT_EQ(key_data[dst_offset + d], key_data_copy[src_offset + d])
                << "Key mismatch at head=" << h << ", dim=" << d;
            EXPECT_FLOAT_EQ(value_data[dst_offset + d], value_data_copy[src_offset + d])
                << "Value mismatch at head=" << h << ", dim=" << d;
        }
    }
}

// Test actual reorder_kv_cache kernel with cross-block copy
TEST(PaKVReorderKernelTest, FloatCacheCrossBlock) {
    using namespace ov::intel_cpu;
    using namespace ov::Extensions::Cpu::XARCH;

    constexpr size_t num_blocks = 4;
    constexpr size_t num_heads = 2;
    constexpr size_t block_size = 32;
    constexpr size_t head_size = 64;

    // Create key/value cache data
    std::vector<float> key_data(num_blocks * num_heads * block_size * head_size);
    std::vector<float> value_data(num_blocks * num_heads * block_size * head_size);

    // Initialize with block and token ID
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t t = 0; t < block_size; t++) {
                for (size_t d = 0; d < head_size; d++) {
                    size_t idx = b * num_heads * block_size * head_size +
                                h * block_size * head_size +
                                t * head_size + d;
                    key_data[idx] = static_cast<float>(b * 1000 + t);
                    value_data[idx] = static_cast<float>(b * 1000 + t + 50000);
                }
            }
        }
    }

    // Save original data
    std::vector<float> key_data_copy = key_data;
    std::vector<float> value_data_copy = value_data;

    // Wrap in PlainTensor
    PlainTensor key_cache;
    key_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, key_data.data());

    PlainTensor value_cache;
    value_cache.resize<float>({num_blocks, num_heads, block_size, head_size}, value_data.data());

    // Copy from block 0, token 5 (logical=5) to block 1, token 10 (logical=42)
    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {5, 42};  // 42 = 32 + 10
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};

    PlainTensor block_indices;
    block_indices.resize<int32_t>({3}, block_indices_data.data());

    PlainTensor block_indices_begins;
    block_indices_begins.resize<int32_t>({2}, block_indices_begins_data.data());

    PlainTensor block_update_indices;
    block_update_indices.resize<int32_t>({2}, block_update_indices_data.data());

    PlainTensor block_update_indices_begins;
    block_update_indices_begins.resize<int32_t>({2}, block_update_indices_begins_data.data());

    // Execute kernel
    auto cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     false,  // key_by_channel
                     false,  // value_by_channel
                     cpu_parallel);

    // Verify: block 1, token 10 should have data from block 0, token 5
    for (size_t h = 0; h < num_heads; h++) {
        size_t src_offset = 0 * num_heads * block_size * head_size + h * block_size * head_size + 5 * head_size;
        size_t dst_offset = 1 * num_heads * block_size * head_size + h * block_size * head_size + 10 * head_size;

        for (size_t d = 0; d < head_size; d++) {
            EXPECT_FLOAT_EQ(key_data[dst_offset + d], key_data_copy[src_offset + d])
                << "Key mismatch at head=" << h << ", dim=" << d;
            EXPECT_FLOAT_EQ(value_data[dst_offset + d], value_data_copy[src_offset + d])
                << "Value mismatch at head=" << h << ", dim=" << d;
        }
    }
}

// Test actual reorder_kv_cache kernel with quantized cache (key by-channel, value by-channel)
TEST(PaKVReorderKernelTest, QuantizedCacheBothByChannel) {
    using namespace ov::intel_cpu;
    using namespace ov::Extensions::Cpu::XARCH;

    constexpr size_t num_blocks = 4;
    constexpr size_t num_heads = 2;
    constexpr size_t block_size = 32;
    constexpr size_t head_size = 64;

    // By-channel quantization: scale/zp shape [num_blocks, num_heads, head_size]
    size_t key_params_bytes = 2 * sizeof(float) * head_size;  // per [block,head]: scale[head_size] + zp[head_size]
    size_t value_params_bytes = 2 * sizeof(float) * head_size;
    size_t key_data_bytes = block_size * head_size;  // uint8 data
    size_t value_data_bytes = block_size * head_size;

    // Total size per block-head: params + data
    size_t key_block_head_bytes = key_params_bytes + key_data_bytes;
    size_t value_block_head_bytes = value_params_bytes + value_data_bytes;

    std::vector<uint8_t> key_cache_data(num_blocks * num_heads * key_block_head_bytes);
    std::vector<uint8_t> value_cache_data(num_blocks * num_heads * value_block_head_bytes);

    // Initialize quantized data with recognizable patterns
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            size_t base_offset = (b * num_heads + h) * key_block_head_bytes;

            float* key_scales = reinterpret_cast<float*>(key_cache_data.data() + base_offset);
            float* key_zps = reinterpret_cast<float*>(key_cache_data.data() + base_offset + sizeof(float) * head_size);

            // Initialize quantized token data first, then calculate proper scale/zp from it
            uint8_t* key_data = key_cache_data.data() + base_offset + key_params_bytes;
            for (size_t t = 0; t < block_size; t++) {
                for (size_t d = 0; d < head_size; d++) {
                    // Use values in valid uint8 range [0, 255]
                    key_data[t * head_size + d] = static_cast<uint8_t>((b * 10 + t) % 256);
                }
            }

            // Calculate scale/zp from the quantized data to ensure dequant->requant round-trip works
            // For uint8: dequant = (quant - zp) * scale
            for (size_t d = 0; d < head_size; d++) {
                key_scales[d] = 0.1f;     // scale
                key_zps[d] = 128.0f;      // uint8 typical zero point
            }

            // Same for value cache
            size_t value_base = (b * num_heads + h) * value_block_head_bytes;
            float* value_scales = reinterpret_cast<float*>(value_cache_data.data() + value_base);
            float* value_zps = reinterpret_cast<float*>(value_cache_data.data() + value_base + sizeof(float) * head_size);

            uint8_t* value_data = value_cache_data.data() + value_base + value_params_bytes;
            for (size_t t = 0; t < block_size; t++) {
                for (size_t d = 0; d < head_size; d++) {
                    value_data[t * head_size + d] = static_cast<uint8_t>((b * 10 + t + 50) % 256);
                }
            }

            for (size_t d = 0; d < head_size; d++) {
                value_scales[d] = 0.1f;
                value_zps[d] = 128.0f;  // uint8 typical zero point
            }
        }
    }

    // Save copy for verification
    std::vector<uint8_t> key_cache_copy = key_cache_data;
    std::vector<uint8_t> value_cache_copy = value_cache_data;

    size_t key_dim2_with_params = block_size + 2 * sizeof(float);  // 32 + 8 = 40
    size_t value_dim2_with_params = block_size + 2 * sizeof(float);

    // Custom strides to match actual memory layout (in uint8 elements)
    size_t key_strides[4] = {
        num_heads * key_block_head_bytes,  // stride[0]: bytes per block
        key_block_head_bytes,               // stride[1]: bytes per head
        head_size,                           // stride[2]: bytes per "token" (params or data)
        1                                    // stride[3]: bytes per element in head_size
    };
    size_t value_strides[4] = {
        num_heads * value_block_head_bytes,
        value_block_head_bytes,
        head_size,
        1
    };

    PlainTensor key_cache;
    key_cache.resize<uint8_t>({num_blocks, num_heads, key_dim2_with_params, head_size},
                               key_cache_data.data(), key_strides);

    PlainTensor value_cache;
    value_cache.resize<uint8_t>({num_blocks, num_heads, value_dim2_with_params, head_size},
                                 value_cache_data.data(), value_strides);

    // Copy from token 10 to token 5 in same block 0 (src > dst, backward copy)
    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {10, 5};  // src=10, dst=5 (backward)
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};  // 1 operation (pair)

    PlainTensor block_indices;
    block_indices.resize<int32_t>({3}, block_indices_data.data());

    PlainTensor block_indices_begins;
    block_indices_begins.resize<int32_t>({2}, block_indices_begins_data.data());

    PlainTensor block_update_indices;
    block_update_indices.resize<int32_t>({2}, block_update_indices_data.data());

    PlainTensor block_update_indices_begins;
    block_update_indices_begins.resize<int32_t>({2}, block_update_indices_begins_data.data());

    // Execute kernel with both by-channel
    auto cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     true,   // key_by_channel
                     true,   // value_by_channel
                     cpu_parallel);

    // Verify: For by-channel quantization, requantize recomputes scale/zp based on block statistics
    // So we verify DEQUANTIZED float values, not raw uint8 values
    for (size_t h = 0; h < num_heads; h++) {
        size_t base_offset = (0 * num_heads + h) * key_block_head_bytes;

        // Get potentially new scale/zp after requantization
        float* key_scales = reinterpret_cast<float*>(key_cache_data.data() + base_offset);
        float* key_zps = reinterpret_cast<float*>(key_cache_data.data() + base_offset + sizeof(float) * head_size);
        float* key_scales_orig = reinterpret_cast<float*>(key_cache_copy.data() + base_offset);
        float* key_zps_orig = reinterpret_cast<float*>(key_cache_copy.data() + base_offset + sizeof(float) * head_size);

        uint8_t* key_data = key_cache_data.data() + base_offset + key_params_bytes;
        uint8_t* key_data_orig = key_cache_copy.data() + base_offset + key_params_bytes;

        // Dequantize and compare float values: float = (quant - zp) * scale
        for (size_t d = 0; d < head_size; d++) {
            float result_float = (key_data[5 * head_size + d] - key_zps[d]) * key_scales[d];
            float expected_float = (key_data_orig[10 * head_size + d] - key_zps_orig[d]) * key_scales_orig[d];

            EXPECT_NEAR(result_float, expected_float, 0.2f)
                << "Key dequantized mismatch at head=" << h << ", dim=" << d;
        }

        // Same for value cache
        size_t value_base = (0 * num_heads + h) * value_block_head_bytes;
        float* value_scales = reinterpret_cast<float*>(value_cache_data.data() + value_base);
        float* value_zps = reinterpret_cast<float*>(value_cache_data.data() + value_base + sizeof(float) * head_size);
        float* value_scales_orig = reinterpret_cast<float*>(value_cache_copy.data() + value_base);
        float* value_zps_orig = reinterpret_cast<float*>(value_cache_copy.data() + value_base + sizeof(float) * head_size);

        uint8_t* value_data = value_cache_data.data() + value_base + value_params_bytes;
        uint8_t* value_data_orig = value_cache_copy.data() + value_base + value_params_bytes;

        for (size_t d = 0; d < head_size; d++) {
            float result_float = (value_data[5 * head_size + d] - value_zps[d]) * value_scales[d];
            float expected_float = (value_data_orig[10 * head_size + d] - value_zps_orig[d]) * value_scales_orig[d];

            EXPECT_NEAR(result_float, expected_float, 0.2f)
                << "Value dequantized mismatch at head=" << h << ", dim=" << d;
        }
    }
}

// Test actual reorder_kv_cache kernel with quantized cache (key by-channel, value by-token)
TEST(PaKVReorderKernelTest, QuantizedCacheKeyByChannelValueByToken) {
    constexpr size_t num_blocks = 4;
    constexpr size_t num_heads = 2;
    constexpr size_t block_size = 32;
    constexpr size_t head_size = 64;

    // Key: by-channel quantization (uint8)
    size_t key_params_bytes = 2 * sizeof(float) * head_size;  // scale[head_size] + zp[head_size]
    size_t key_data_bytes = block_size * head_size;  // uint8 data
    size_t key_block_head_bytes = key_params_bytes + key_data_bytes;

    // Value: by-token quantization (INTERLEAVED layout)
    // Each token: [scale(f32), zp(f32), data[head_size](uint8)]
    size_t value_token_stride = 2 * sizeof(float) + head_size;  // bytes per token
    size_t value_block_head_bytes = block_size * value_token_stride;

    std::vector<uint8_t> key_cache_data(num_blocks * num_heads * key_block_head_bytes);
    std::vector<uint8_t> value_cache_data(num_blocks * num_heads * value_block_head_bytes);

    // Initialize key cache (by-channel)
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            size_t base_offset = (b * num_heads + h) * key_block_head_bytes;

            // Scale/zp: by-channel, shared across all tokens
            float* key_scales = reinterpret_cast<float*>(key_cache_data.data() + base_offset);
            float* key_zps = reinterpret_cast<float*>(key_cache_data.data() + base_offset + sizeof(float) * head_size);
            for (size_t d = 0; d < head_size; d++) {
                key_scales[d] = 0.1f;
                key_zps[d] = 128.0f;  // uint8 typical zero point
            }

            // Quantized token data (uint8)
            uint8_t* key_data = key_cache_data.data() + base_offset + key_params_bytes;
            for (size_t t = 0; t < block_size; t++) {
                for (size_t d = 0; d < head_size; d++) {
                    key_data[t * head_size + d] = static_cast<uint8_t>((b * 1000 + t) % 256);
                }
            }
        }
    }

    // Initialize value cache (by-token): INTERLEAVED layout!
    // For by-token quantization, layout is: [scale_0, zp_0, data_0[head_size], scale_1, zp_1, data_1[head_size], ...]
    for (size_t b = 0; b < num_blocks; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            size_t base_offset = (b * num_heads + h) * value_block_head_bytes;
            uint8_t* block_ptr = value_cache_data.data() + base_offset;

            // Each token has: [scale(f32), zp(f32), data[head_size](uint8)]
            size_t token_stride = 2 * sizeof(float) + head_size;  // bytes per token

            for (size_t t = 0; t < block_size; t++) {
                uint8_t* token_ptr = block_ptr + t * token_stride;

                // Write scale and zp for this token
                float* scale_ptr = reinterpret_cast<float*>(token_ptr);
                float* zp_ptr = reinterpret_cast<float*>(token_ptr + sizeof(float));
                *scale_ptr = 0.1f;
                *zp_ptr = 128.0f;  // uint8 typical zero point

                // Write quantized data for this token (uint8)
                uint8_t* data_ptr = token_ptr + 2 * sizeof(float);
                for (size_t d = 0; d < head_size; d++) {
                    data_ptr[d] = static_cast<uint8_t>((b * 10 + t + 50) % 256);
                }
            }
        }
    }

    // Save copy for verification
    std::vector<uint8_t> key_cache_copy = key_cache_data;
    std::vector<uint8_t> value_cache_copy = value_cache_data;

    size_t key_dim2_with_params = block_size + 2 * sizeof(float);  // 32 + 8 = 40

    // Custom strides to match actual memory layout (in uint8 elements)
    size_t key_strides[4] = {
        num_heads * key_block_head_bytes,  // stride[0]: bytes per block
        key_block_head_bytes,               // stride[1]: bytes per head
        head_size,                           // stride[2]: bytes per "token" (params or data)
        1                                    // stride[3]: bytes per element in head_size
    };

    // Value (by-token INTERLEAVED): [num_blocks, num_heads, block_size, head_size+params]
    //   Each token has [scale, zp, data[head_size]] interleaved
    //   dims[3] includes params per token: head_size + 2*sizeof(float)
    //   stride[2] is the byte stride between tokens
    size_t value_dim3_with_params = head_size + 2 * sizeof(float);  // 64 + 8 = 72
    size_t value_strides[4] = {
        num_heads * value_block_head_bytes,  // stride[0]: bytes per block
        value_block_head_bytes,               // stride[1]: bytes per head
        value_token_stride,                   // stride[2]: bytes per token (interleaved)
        1                                     // stride[3]: bytes per element
    };

    PlainTensor key_cache;
    key_cache.resize<uint8_t>({num_blocks, num_heads, key_dim2_with_params, head_size},
                               key_cache_data.data(), key_strides);

    PlainTensor value_cache;
    value_cache.resize<uint8_t>({num_blocks, num_heads, block_size, value_dim3_with_params},
                                 value_cache_data.data(), value_strides);

    // Copy from token 10 to token 5 in same block 0 (src > dst, backward copy)
    std::vector<int32_t> block_indices_data = {0, 1, 2};
    std::vector<int32_t> block_indices_begins_data = {0, 3};
    std::vector<int32_t> block_update_indices_data = {10, 5};  // src=10, dst=5 (backward)
    std::vector<int32_t> block_update_indices_begins_data = {0, 1};  // 1 operation (pair)

    PlainTensor block_indices;
    block_indices.resize<int32_t>({3}, block_indices_data.data());

    PlainTensor block_indices_begins;
    block_indices_begins.resize<int32_t>({2}, block_indices_begins_data.data());

    PlainTensor block_update_indices;
    block_update_indices.resize<int32_t>({2}, block_update_indices_data.data());

    PlainTensor block_update_indices_begins;
    block_update_indices_begins.resize<int32_t>({2}, block_update_indices_begins_data.data());

    // Execute kernel with key by-channel, value by-token
    auto cpu_parallel = std::make_shared<CpuParallel>(ov::intel_cpu::TbbPartitioner::STATIC);
    reorder_kv_cache(key_cache,
                     value_cache,
                     block_indices,
                     block_indices_begins,
                     block_update_indices,
                     block_update_indices_begins,
                     true,   // key_by_channel
                     false,  // value_by_token
                     cpu_parallel);

    // Verify key: by-channel quantization with requantization
    // Compare dequantized values since raw uint8 values change after requantization
    for (size_t h = 0; h < num_heads; h++) {
        size_t base_offset = (0 * num_heads + h) * key_block_head_bytes;

        // Get scales/zps after reorder (may have been recomputed)
        float* key_scales = reinterpret_cast<float*>(key_cache_data.data() + base_offset);
        float* key_zps = reinterpret_cast<float*>(key_cache_data.data() + base_offset + sizeof(float) * head_size);

        // Get scales/zps from original (before reorder)
        float* key_scales_orig = reinterpret_cast<float*>(key_cache_copy.data() + base_offset);
        float* key_zps_orig = reinterpret_cast<float*>(key_cache_copy.data() + base_offset + sizeof(float) * head_size);

        // Get quantized data
        uint8_t* key_data = key_cache_data.data() + base_offset + key_params_bytes;
        uint8_t* key_data_orig = key_cache_copy.data() + base_offset + key_params_bytes;

        // Compare dequantized values: dequant(dst_token5) ≈ dequant(src_token10_orig)
        for (size_t d = 0; d < head_size; d++) {
            // Dequantize dst token 5 with current scale/zp
            float dst_dequant = (key_data[5 * head_size + d] - key_zps[d]) * key_scales[d];

            // Dequantize src token 10 from original with original scale/zp
            float src_dequant = (key_data_orig[10 * head_size + d] - key_zps_orig[d]) * key_scales_orig[d];

            EXPECT_NEAR(dst_dequant, src_dequant, 0.2f)
                << "Key dequantized mismatch at head=" << h << ", dim=" << d;
        }
    }

    // Verify value: by-token INTERLEAVED layout, direct memcpy of [scale, zp, data] per token
    for (size_t h = 0; h < num_heads; h++) {
        size_t base_offset = (0 * num_heads + h) * value_block_head_bytes;
        uint8_t* block_ptr = value_cache_data.data() + base_offset;
        uint8_t* block_ptr_orig = value_cache_copy.data() + base_offset;

        // Access dst token 5 (should have src token 10's data)
        uint8_t* dst_token_ptr = block_ptr + 5 * value_token_stride;
        uint8_t* src_token_ptr = block_ptr_orig + 10 * value_token_stride;

        // Check scale/zp copied correctly (interleaved)
        float* dst_scale = reinterpret_cast<float*>(dst_token_ptr);
        float* dst_zp = reinterpret_cast<float*>(dst_token_ptr + sizeof(float));
        float* src_scale = reinterpret_cast<float*>(src_token_ptr);
        float* src_zp = reinterpret_cast<float*>(src_token_ptr + sizeof(float));

        EXPECT_FLOAT_EQ(*dst_scale, *src_scale)
            << "Value scale mismatch at head=" << h;
        EXPECT_FLOAT_EQ(*dst_zp, *src_zp)
            << "Value zp mismatch at head=" << h;

        // Check quantized data copied correctly (uint8)
        uint8_t* dst_data = dst_token_ptr + 2 * sizeof(float);
        uint8_t* src_data = src_token_ptr + 2 * sizeof(float);

        for (size_t d = 0; d < head_size; d++) {
            EXPECT_EQ(dst_data[d], src_data[d])
                << "Value data mismatch at head=" << h << ", dim=" << d;
        }
    }
}
