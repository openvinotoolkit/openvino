// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <numeric>

#include "infer_request_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "util.hpp"

namespace {

// Helper: create an ov::SoPtr<ov::ITensor> with shape [1, seq_len, num_layers, proj_dim]
// and fill with sequential float values starting from `start_val`.
ov::SoPtr<ov::ITensor> make_per_layer_tensor(size_t seq_len,
                                             size_t num_layers,
                                             size_t proj_dim,
                                             float start_val = 0.f) {
    ov::Shape shape{1, seq_len, num_layers, proj_dim};
    auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape));
    auto* data = reinterpret_cast<float*>(tensor->data());
    for (size_t i = 0; i < tensor->get_size(); ++i) {
        data[i] = start_val + static_cast<float>(i);
    }
    return tensor;
}

// Helper: read all floats from a tensor into a vector.
std::vector<float> to_vec(const ov::SoPtr<ov::ITensor>& t) {
    const auto* data = reinterpret_cast<const float*>(t->data());
    return std::vector<float>(data, data + t->get_size());
}

// --- copy_per_layer_inputs_chunk_to_left tests ----------------------------------

// Test 1: copy the first chunk of src to dst.
// src shape [1,4,2,2], dst shape [1,2,2,2].
// Copy chunk_tokens=2 starting at src_offset=0 -> dst should equal src[0:2].
TEST(PerLayerInputsCopyTest, ChunkAtOffsetZeroCopiesToLeft) {
    // src: [1, 4, 2, 2], values 0..15
    auto src = make_per_layer_tensor(/*seq_len=*/4, /*num_layers=*/2, /*proj_dim=*/2, /*start_val=*/0.f);
    // dst: [1, 2, 2, 2]
    auto dst = make_per_layer_tensor(/*seq_len=*/2, /*num_layers=*/2, /*proj_dim=*/2, /*start_val=*/99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/0, /*chunk=*/2));

    // src tokens 0,1 -> values [0..7]
    const auto result = to_vec(dst);
    // dst is left-aligned; since chunk==dst_seq_len the entire dst is overwritten
    std::vector<float> expected = {0.f,
                                   1.f,
                                   2.f,
                                   3.f,  // token 0
                                   4.f,
                                   5.f,
                                   6.f,
                                   7.f};  // token 1
    EXPECT_EQ(result, expected);
}

// Test 2: copy a middle chunk (offset=2, chunk=2) from a src with 6 tokens.
// dst has 4 token slots; chunk fills the left 2 slots and leaves the trailing slots unchanged.
TEST(PerLayerInputsCopyTest, ChunkAtOffsetCopiesLeftAligned) {
    // src: [1, 6, 2, 2], values 0..23
    auto src = make_per_layer_tensor(6, 2, 2, 0.f);
    // dst: [1, 4, 2, 2], sequential values starting from 99 (99, 100, 101, ...)
    auto dst = make_per_layer_tensor(4, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/2, /*chunk=*/2));

    // src tokens at offset 2,3 -> src flat indices [8..15]
    const auto result = to_vec(dst);
    // Left-aligned: dst tokens 0,1 hold src[2],src[3].
    std::vector<float> expected = {8.f,
                                   9.f,
                                   10.f,
                                   11.f,  // token 0
                                   12.f,
                                   13.f,
                                   14.f,
                                   15.f,  // token 1
                                   107.f,
                                   108.f,
                                   109.f,
                                   110.f,  // unchanged token 2
                                   111.f,
                                   112.f,
                                   113.f,
                                   114.f};  // unchanged token 3
    EXPECT_EQ(result, expected);
}

// Test 3: chunk_tokens == 1 (generate step).
TEST(PerLayerInputsCopyTest, SingleTokenChunkCopiesToFirstSlot) {
    // src: [1, 3, 2, 2], values 0..11
    auto src = make_per_layer_tensor(3, 2, 2, 0.f);
    // dst: [1, 1, 2, 2]
    auto dst = make_per_layer_tensor(1, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/0, /*chunk=*/1));

    const auto result = to_vec(dst);
    // src token 0 -> values [0,1,2,3]
    std::vector<float> expected = {0.f, 1.f, 2.f, 3.f};
    EXPECT_EQ(result, expected);
}

// Test 4: chunk_tokens == 0 must throw.
TEST(PerLayerInputsCopyTest, ZeroChunkTokensThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, 0, 0));
}

// Test 5: offset beyond src seq_len must throw.
TEST(PerLayerInputsCopyTest, OffsetExceedsSrcSeqLenThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/5, /*chunk=*/1));
}

// Test 6: offset+chunk exceeds src seq_len must throw.
TEST(PerLayerInputsCopyTest, ChunkRangeExceedsSrcSeqLenThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/3, /*chunk=*/2));
}

// Test 7: chunk_tokens > dst_seq_len must throw.
TEST(PerLayerInputsCopyTest, ChunkExceedsDstSeqLenThrows) {
    auto src = make_per_layer_tensor(8, 2, 2);
    auto dst = make_per_layer_tensor(2, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/0, /*chunk=*/4));
}

// Test 8: src and dst have different per-token byte sizes (different num_layers) must throw.
TEST(PerLayerInputsCopyTest, PerTokenByteMismatchThrows) {
    auto src = make_per_layer_tensor(4, /*num_layers=*/2, 2);
    auto dst = make_per_layer_tensor(4, /*num_layers=*/3, 2);  // different num_layers
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_left(src, dst, /*offset=*/0, /*chunk=*/2));
}

// --- copy_to_left for whole-prefill/generate paths ------------------------------

// Test 9: copy_to_left writes src into the left end of dst; trailing bytes are left unchanged.
TEST(PerLayerInputsCopyTest, CopyToLeftLeavesTrailingBytesUnchanged) {
    // src: [1, 2, 2, 2], values 0..7
    auto src = make_per_layer_tensor(2, 2, 2, 0.f);
    // dst: [1, 4, 2, 2], sequential values starting from 99 (99, 100, 101, ...)
    auto dst = make_per_layer_tensor(4, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_to_left(src, dst));

    const auto result = to_vec(dst);
    std::vector<float> expected = {0.f,
                                   1.f,
                                   2.f,
                                   3.f,  // src token 0
                                   4.f,
                                   5.f,
                                   6.f,
                                   7.f,  // src token 1
                                   107.f,
                                   108.f,
                                   109.f,
                                   110.f,  // unchanged (token 2)
                                   111.f,
                                   112.f,
                                   113.f,
                                   114.f};  // unchanged (token 3)
    EXPECT_EQ(result, expected);
}

// Test 10: copy_to_left when src size == dst size copies everything.
TEST(PerLayerInputsCopyTest, CopyToLeftSameSizeCopiesAll) {
    auto src = make_per_layer_tensor(2, 2, 2, 1.f);
    auto dst = make_per_layer_tensor(2, 2, 2, 0.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_to_left(src, dst));

    EXPECT_EQ(to_vec(dst), to_vec(src));
}

TEST(InferRequestUtilsTest, PositionIdsAreLeftAlignedForEveryPlane) {
    ov::Tensor source(ov::element::i64, ov::Shape{3, 1, 2});
    ov::Tensor destination(ov::element::i64, ov::Shape{3, 1, 4});
    std::iota(source.data<int64_t>(), source.data<int64_t>() + source.get_size(), 1);
    std::fill_n(destination.data<int64_t>(), destination.get_size(), 99);

    ov::npuw::util::pad_position_ids(ov::get_tensor_impl(destination), ov::get_tensor_impl(source));

    const std::vector<int64_t> expected{1, 2, 99, 99, 3, 4, 99, 99, 5, 6, 99, 99};
    EXPECT_EQ(std::vector<int64_t>(destination.data<int64_t>(), destination.data<int64_t>() + destination.get_size()),
              expected);
}

}  // namespace
