// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "infer_request_utils.hpp"
#include "kv_cache_sliding_window_manager.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::test::npuw {

namespace {

namespace uu = ov::npuw::util;

ov::SoPtr<ov::ITensor> make_cpu_tensor(const ov::Shape& shape) {
    return ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape));
}

// [1, heads, seq_len, emb] for kv_dim == 2, [1, heads, emb, seq_len] for kv_dim == 3.
ov::Shape kv_shape(uint32_t kv_dim, uint32_t seq_len, uint32_t heads = 2u, uint32_t emb = 3u) {
    return (kv_dim == 3u) ? ov::Shape{1u, heads, emb, seq_len} : ov::Shape{1u, heads, seq_len, emb};
}

// Generic element-wise walk over a (possibly non-contiguous / ROI) f32 tensor, visiting
// every logical element in row-major order. Works regardless of kv_dim/layout, so it can
// fill or read back a single-token slice (which is only contiguous for some layouts).
template <typename Fn>
void for_each_element(const ov::SoPtr<ov::ITensor>& tensor, Fn&& fn) {
    const auto& shape = tensor->get_shape();
    const auto& strides = tensor->get_strides();  // byte strides
    auto* base = static_cast<uint8_t*>(tensor->data());
    std::vector<size_t> idx(shape.size(), 0u);
    const size_t total = tensor->get_size();
    for (size_t linear = 0; linear < total; ++linear) {
        size_t byte_offset = 0u;
        for (size_t d = 0; d < shape.size(); ++d) {
            byte_offset += idx[d] * strides[d];
        }
        fn(*reinterpret_cast<float*>(base + byte_offset));
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) {
                break;
            }
            idx[d] = 0u;
        }
    }
}

// Fills physical column `pos` along `kv_dim` with a single repeated value, so the token's
// identity can be recovered later via a single read (and cross-checked via
// expect_token_value below).
void write_token_value(const ov::SoPtr<ov::ITensor>& tensor, uint32_t kv_dim, uint32_t pos, float value) {
    auto slice = uu::make_tensor_slice(tensor, kv_dim, pos, pos + 1);
    for_each_element(slice, [&](float& v) {
        v = value;
    });
}

// Asserts every element of physical column `pos` along `kv_dim` equals `value`.
void expect_token_value(const ov::SoPtr<ov::ITensor>& tensor, uint32_t kv_dim, uint32_t pos, float value) {
    auto slice = uu::make_tensor_slice(tensor, kv_dim, pos, pos + 1);
    for_each_element(slice, [&](float& v) {
        EXPECT_FLOAT_EQ(v, value) << "kv_dim=" << kv_dim << " pos=" << pos;
    });
}

// Builds a source tensor of `count` tokens along `kv_dim`, whose logical token `i` (i.e.
// physical column i, since a freshly-produced source is always contiguous/right-aligned)
// is filled with `first_value + i`.
ov::SoPtr<ov::ITensor> make_src_tokens(uint32_t kv_dim, uint32_t count, float first_value) {
    auto src = make_cpu_tensor(kv_shape(kv_dim, count));
    for (uint32_t i = 0; i < count; ++i) {
        write_token_value(src, kv_dim, i, first_value + static_cast<float>(i));
    }
    return src;
}

}  // namespace

class WriteKvSliceSlidingTest : public ::testing::TestWithParam<uint32_t> {};

// kv_dim is parameterized: 2 (standard [1,H,S,E] layout) and 3 (transposed-V layout).
INSTANTIATE_TEST_SUITE_P(KvDims, WriteKvSliceSlidingTest, ::testing::Values(2u, 3u));

TEST_P(WriteKvSliceSlidingTest, CircularWarmupMatchesLeftAligned) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst_left = make_cpu_tensor(kv_shape(kv_dim, capacity));
    auto dst_circ = make_cpu_tensor(kv_shape(kv_dim, capacity));
    auto src = make_src_tokens(kv_dim, 5u, /*first_value=*/100.f);

    uu::write_kv_slice_sliding(dst_left, src, kv_dim, kv_dim, /*num_stored_tokens_before=*/0u, /*num_new_tokens=*/5u,
                               uu::SlidingBufferLayout::LeftAligned);
    uu::write_kv_slice_sliding(dst_circ, src, kv_dim, kv_dim, /*num_stored_tokens_before=*/0u, /*num_new_tokens=*/5u,
                               uu::SlidingBufferLayout::Circular);

    for (uint32_t i = 0; i < 5u; ++i) {
        expect_token_value(dst_left, kv_dim, i, 100.f + static_cast<float>(i));
        expect_token_value(dst_circ, kv_dim, i, 100.f + static_cast<float>(i));
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularWrapsToStartWithoutSplit) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // Pretend absolute positions [0..7] are already stored: physical slot i holds
    // value (1000 + i).
    for (uint32_t i = 0; i < capacity; ++i) {
        write_token_value(dst, kv_dim, i, 1000.f + static_cast<float>(i));
    }

    // One new token arrives at absolute position 8 -> physical slot (8 % 8) == 0.
    auto src = make_src_tokens(kv_dim, 1u, /*first_value=*/2000.f);
    uu::write_kv_slice_sliding(dst, src, kv_dim, kv_dim, /*num_stored_tokens_before=*/capacity, /*num_new_tokens=*/1u,
                               uu::SlidingBufferLayout::Circular);

    expect_token_value(dst, kv_dim, 0u, 2000.f);
    for (uint32_t i = 1; i < capacity; ++i) {
        expect_token_value(dst, kv_dim, i, 1000.f + static_cast<float>(i));
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularWriteSplitsAcrossWrapBoundary) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // Physical layout represents logical positions [8,9,10,11,12,5,6,7] at physical
    // slots [0,1,2,3,4,5,6,7] respectively (i.e. 13 tokens (0..12) already written).
    const std::vector<float> initial = {2000.f, 2001.f, 2002.f, 2003.f, 2004.f, 1005.f, 1006.f, 1007.f};
    for (uint32_t i = 0; i < capacity; ++i) {
        write_token_value(dst, kv_dim, i, initial[i]);
    }

    // 6 new tokens arrive in one call, absolute positions [13..18].
    // dst_start = 13 % 8 = 5, tokens_to_write = 6 -> wraps (5 + 6 > 8):
    //   leg 1: physical [5,8)  <- src tokens [0,1,2] (positions 13,14,15)
    //   leg 2: physical [0,3)  <- src tokens [3,4,5] (positions 16,17,18)
    auto src = make_src_tokens(kv_dim, 6u, /*first_value=*/3013.f);
    uu::write_kv_slice_sliding(dst, src, kv_dim, kv_dim, /*num_stored_tokens_before=*/13u, /*num_new_tokens=*/6u,
                               uu::SlidingBufferLayout::Circular);

    const std::vector<float> expected = {3016.f, 3017.f, 3018.f, 2003.f, 2004.f, 3013.f, 3014.f, 3015.f};
    for (uint32_t i = 0; i < capacity; ++i) {
        expect_token_value(dst, kv_dim, i, expected[i]);
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularChunkLargerThanCapacityKeepsOnlyNewestTail) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 4u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // A single chunked-prefill-style call writes 10 tokens (absolute positions 0..9)
    // against an empty, capacity-4 buffer in one shot. Only the newest 4 (positions
    // 6,7,8,9) survive; dst_start = (0 + (10-4)) % 4 = 2, wraps (2+4>4):
    //   leg 1: physical [2,4) <- src tokens [0,1] (positions 6,7)
    //   leg 2: physical [0,2) <- src tokens [2,3] (positions 8,9)
    auto src = make_src_tokens(kv_dim, 10u, /*first_value=*/0.f);
    uu::write_kv_slice_sliding(dst, src, kv_dim, kv_dim, /*num_stored_tokens_before=*/0u, /*num_new_tokens=*/10u,
                               uu::SlidingBufferLayout::Circular);

    const std::vector<float> expected = {8.f, 9.f, 6.f, 7.f};
    for (uint32_t i = 0; i < capacity; ++i) {
        expect_token_value(dst, kv_dim, i, expected[i]);
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularAndLeftAlignedHoldSameLogicalContentOverManySteps) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 5u;

    auto dst_left = make_cpu_tensor(kv_shape(kv_dim, capacity));
    auto dst_circ = make_cpu_tensor(kv_shape(kv_dim, capacity));

    uint32_t stored = 0u;
    for (uint32_t step = 0; step < 20u; ++step) {
        // One new token per step, value == its own absolute position (so recovering it
        // later is a direct identity check).
        auto src = make_src_tokens(kv_dim, 1u, /*first_value=*/static_cast<float>(stored));
        uu::write_kv_slice_sliding(dst_left, src, kv_dim, kv_dim, stored, 1u, uu::SlidingBufferLayout::LeftAligned);
        uu::write_kv_slice_sliding(dst_circ, src, kv_dim, kv_dim, stored, 1u, uu::SlidingBufferLayout::Circular);
        stored += 1u;

        const uint32_t valid = std::min(stored, capacity);
        const uint32_t window_start = stored - valid;  // oldest absolute position still in window
        for (uint32_t rank = 0; rank < valid; ++rank) {
            const uint32_t abs_pos = window_start + rank;
            // LeftAligned: rank-th oldest surviving token sits at physical index `rank`.
            expect_token_value(dst_left, kv_dim, rank, static_cast<float>(abs_pos));
            // Circular: token at abs_pos always sits at physical index (abs_pos % capacity).
            expect_token_value(dst_circ, kv_dim, abs_pos % capacity, static_cast<float>(abs_pos));
        }
    }
}

}  // namespace ov::test::npuw
