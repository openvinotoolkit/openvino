// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "infer_request_utils.hpp"
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

// Walk all logical f32 elements of a tensor (including non-contiguous views).
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

// Fill one token column at `pos` along `kv_dim` with `value`.
void write_token_value(const ov::SoPtr<ov::ITensor>& tensor, uint32_t kv_dim, uint32_t pos, float value) {
    auto slice = uu::make_tensor_slice(tensor, kv_dim, pos, pos + 1);
    for_each_element(slice, [&](float& v) {
        v = value;
    });
}

// Assert one token column at `pos` along `kv_dim` equals `value`.
void expect_token_value(const ov::SoPtr<ov::ITensor>& tensor, uint32_t kv_dim, uint32_t pos, float value) {
    auto slice = uu::make_tensor_slice(tensor, kv_dim, pos, pos + 1);
    for_each_element(slice, [&](float& v) {
        EXPECT_FLOAT_EQ(v, value) << "kv_dim=" << kv_dim << " pos=" << pos;
    });
}

// Build contiguous source tokens with values [first_value, first_value+count).
ov::SoPtr<ov::ITensor> make_src_tokens(uint32_t kv_dim, uint32_t count, float first_value) {
    auto src = make_cpu_tensor(kv_shape(kv_dim, count));
    for (uint32_t i = 0; i < count; ++i) {
        write_token_value(src, kv_dim, i, first_value + static_cast<float>(i));
    }
    return src;
}

void expect_cross_layout_write(uint32_t src_kv_dim, uint32_t dst_kv_dim, bool circular_write) {
    const uint32_t capacity = 6u;
    auto dst = make_cpu_tensor(kv_shape(dst_kv_dim, capacity));
    auto src = make_src_tokens(src_kv_dim, 3u, /*first_value=*/10.f);

    if (circular_write) {
        uu::write_swa_kv_slice_circular(dst,
                                        src,
                                        dst_kv_dim,
                                        src_kv_dim,
                                        /*num_stored_tokens_before=*/0u,
                                        /*num_new_tokens=*/3u);
    } else {
        uu::write_swa_kv_slice_left_aligned(dst,
                                            src,
                                            dst_kv_dim,
                                            src_kv_dim,
                                            /*num_stored_tokens_before=*/0u,
                                            /*num_new_tokens=*/3u);
    }

    for (uint32_t i = 0; i < 3u; ++i) {
        expect_token_value(dst, dst_kv_dim, i, 10.f + static_cast<float>(i));
    }
}

}  // namespace

class WriteKvSliceSlidingTest : public ::testing::TestWithParam<uint32_t> {};

// kv_dim is parameterized: 2 ([1,H,S,E]) and 3 ([1,H,E,S]).
INSTANTIATE_TEST_SUITE_P(KvDims, WriteKvSliceSlidingTest, ::testing::Values(2u, 3u));

TEST_P(WriteKvSliceSlidingTest, CircularWarmupMatchesLeftAligned) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst_left = make_cpu_tensor(kv_shape(kv_dim, capacity));
    auto dst_circ = make_cpu_tensor(kv_shape(kv_dim, capacity));
    auto src = make_src_tokens(kv_dim, 5u, /*first_value=*/100.f);

    uu::write_swa_kv_slice_left_aligned(dst_left,
                                        src,
                                        kv_dim,
                                        kv_dim,
                                        /*num_stored_tokens_before=*/0u,
                                        /*num_new_tokens=*/5u);
    uu::write_swa_kv_slice_circular(dst_circ,
                                    src,
                                    kv_dim,
                                    kv_dim,
                                    /*num_stored_tokens_before=*/0u,
                                    /*num_new_tokens=*/5u);

    for (uint32_t i = 0; i < 5u; ++i) {
        expect_token_value(dst_left, kv_dim, i, 100.f + static_cast<float>(i));
        expect_token_value(dst_circ, kv_dim, i, 100.f + static_cast<float>(i));
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularWrapsToStartWithoutSplit) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // Seed slots [0..7] with distinct values.
    for (uint32_t i = 0; i < capacity; ++i) {
        write_token_value(dst, kv_dim, i, 1000.f + static_cast<float>(i));
    }

    // New token at absolute position 8 overwrites slot 0.
    auto src = make_src_tokens(kv_dim, 1u, /*first_value=*/2000.f);
    uu::write_swa_kv_slice_circular(dst,
                                    src,
                                    kv_dim,
                                    kv_dim,
                                    /*num_stored_tokens_before=*/capacity,
                                    /*num_new_tokens=*/1u);

    expect_token_value(dst, kv_dim, 0u, 2000.f);
    for (uint32_t i = 1; i < capacity; ++i) {
        expect_token_value(dst, kv_dim, i, 1000.f + static_cast<float>(i));
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularWriteSplitsAcrossWrapBoundary) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 8u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // Seed a pre-wrapped layout with 13 prior writes.
    const std::vector<float> initial = {2000.f, 2001.f, 2002.f, 2003.f, 2004.f, 1005.f, 1006.f, 1007.f};
    for (uint32_t i = 0; i < capacity; ++i) {
        write_token_value(dst, kv_dim, i, initial[i]);
    }

    // 6-token write wraps at slot 5 and splits into tail+head legs.
    auto src = make_src_tokens(kv_dim, 6u, /*first_value=*/3013.f);
    uu::write_swa_kv_slice_circular(dst,
                                    src,
                                    kv_dim,
                                    kv_dim,
                                    /*num_stored_tokens_before=*/13u,
                                    /*num_new_tokens=*/6u);

    const std::vector<float> expected = {3016.f, 3017.f, 3018.f, 2003.f, 2004.f, 3013.f, 3014.f, 3015.f};
    for (uint32_t i = 0; i < capacity; ++i) {
        expect_token_value(dst, kv_dim, i, expected[i]);
    }
}

TEST_P(WriteKvSliceSlidingTest, CircularChunkLargerThanCapacityKeepsOnlyNewestTail) {
    const uint32_t kv_dim = GetParam();
    const uint32_t capacity = 4u;

    auto dst = make_cpu_tensor(kv_shape(kv_dim, capacity));
    // Writing 10 tokens into capacity 4 keeps only newest tail [6,7,8,9].
    auto src = make_src_tokens(kv_dim, 10u, /*first_value=*/0.f);
    uu::write_swa_kv_slice_circular(dst,
                                    src,
                                    kv_dim,
                                    kv_dim,
                                    /*num_stored_tokens_before=*/0u,
                                    /*num_new_tokens=*/10u);

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
        // One new token per step; value equals absolute position.
        auto src = make_src_tokens(kv_dim, 1u, /*first_value=*/static_cast<float>(stored));
        uu::write_swa_kv_slice_left_aligned(dst_left, src, kv_dim, kv_dim, stored, 1u);
        uu::write_swa_kv_slice_circular(dst_circ, src, kv_dim, kv_dim, stored, 1u);
        stored += 1u;

        const uint32_t valid = std::min(stored, capacity);
        const uint32_t window_start = stored - valid;  // oldest absolute position still in window
        for (uint32_t rank = 0; rank < valid; ++rank) {
            const uint32_t abs_pos = window_start + rank;
            // LeftAligned keeps oldest-surviving tokens packed from index 0.
            expect_token_value(dst_left, kv_dim, rank, static_cast<float>(abs_pos));
            // Circular keeps token at physical slot abs_pos % capacity.
            expect_token_value(dst_circ, kv_dim, abs_pos % capacity, static_cast<float>(abs_pos));
        }
    }
}

TEST(WriteKvSliceSlidingCrossLayoutTest, CircularConvertsKvDim2To3) {
    expect_cross_layout_write(/*src_kv_dim=*/2u, /*dst_kv_dim=*/3u, /*circular_write=*/true);
}

TEST(WriteKvSliceSlidingCrossLayoutTest, CircularConvertsKvDim3To2) {
    expect_cross_layout_write(/*src_kv_dim=*/3u, /*dst_kv_dim=*/2u, /*circular_write=*/true);
}

TEST(WriteKvSliceSlidingCrossLayoutTest, LeftAlignedConvertsKvDim2To3) {
    expect_cross_layout_write(/*src_kv_dim=*/2u, /*dst_kv_dim=*/3u, /*circular_write=*/false);
}

TEST(WriteKvSliceSlidingCrossLayoutTest, LeftAlignedConvertsKvDim3To2) {
    expect_cross_layout_write(/*src_kv_dim=*/3u, /*dst_kv_dim=*/2u, /*circular_write=*/false);
}

}  // namespace ov::test::npuw
