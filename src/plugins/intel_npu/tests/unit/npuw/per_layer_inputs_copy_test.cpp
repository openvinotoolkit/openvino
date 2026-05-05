// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

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

// --- copy_per_layer_inputs_chunk_to_right tests ---------------------------------

// Test 1: copy the first chunk of src to dst.
// src shape [1,4,2,2], dst shape [1,2,2,2].
// Copy chunk_tokens=2 starting at src_offset=0 -> dst should equal src[0:2].
TEST(PerLayerInputsCopyTest, ChunkAtOffsetZeroCopiesToRight) {
    // src: [1, 4, 2, 2], values 0..15
    auto src = make_per_layer_tensor(/*seq_len=*/4, /*num_layers=*/2, /*proj_dim=*/2, /*start_val=*/0.f);
    // dst: [1, 2, 2, 2]
    auto dst = make_per_layer_tensor(/*seq_len=*/2, /*num_layers=*/2, /*proj_dim=*/2, /*start_val=*/99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/0, /*chunk=*/2));

    // src tokens 0,1 -> values [0..7]
    const auto result = to_vec(dst);
    // dst is right-aligned; since chunk==dst_seq_len the entire dst is overwritten
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
// dst has 4 token slots; chunk fills the right 2 slots, left 2 slots are left unchanged.
TEST(PerLayerInputsCopyTest, ChunkAtOffsetCopiesRightAlignedLeavesLeadingBytesUnchanged) {
    // src: [1, 6, 2, 2], values 0..23
    auto src = make_per_layer_tensor(6, 2, 2, 0.f);
    // dst: [1, 4, 2, 2], sequential values starting from 99 (99, 100, 101, ...)
    auto dst = make_per_layer_tensor(4, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/2, /*chunk=*/2));

    // src tokens at offset 2,3 -> src flat indices [8..15]
    const auto result = to_vec(dst);
    // Right-aligned: dst tokens 0,1 are unchanged (sequential from start_val=99); dst tokens 2,3 hold src[2],src[3]
    std::vector<float> expected = {99.f,
                                   100.f,
                                   101.f,
                                   102.f,  // unchanged (token 0)
                                   103.f,
                                   104.f,
                                   105.f,
                                   106.f,  // unchanged (token 1)
                                   8.f,
                                   9.f,
                                   10.f,
                                   11.f,  // token 2
                                   12.f,
                                   13.f,
                                   14.f,
                                   15.f};  // token 3
    EXPECT_EQ(result, expected);
}

// Test 3: chunk_tokens == 1 (generate step).
TEST(PerLayerInputsCopyTest, SingleTokenChunkCopiesToLastSlot) {
    // src: [1, 3, 2, 2], values 0..11
    auto src = make_per_layer_tensor(3, 2, 2, 0.f);
    // dst: [1, 1, 2, 2]
    auto dst = make_per_layer_tensor(1, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/0, /*chunk=*/1));

    const auto result = to_vec(dst);
    // src token 0 -> values [0,1,2,3]
    std::vector<float> expected = {0.f, 1.f, 2.f, 3.f};
    EXPECT_EQ(result, expected);
}

// Test 4: chunk_tokens == 0 must throw.
TEST(PerLayerInputsCopyTest, ZeroChunkTokensThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, 0, 0));
}

// Test 5: offset beyond src seq_len must throw.
TEST(PerLayerInputsCopyTest, OffsetExceedsSrcSeqLenThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/5, /*chunk=*/1));
}

// Test 6: offset+chunk exceeds src seq_len must throw.
TEST(PerLayerInputsCopyTest, ChunkRangeExceedsSrcSeqLenThrows) {
    auto src = make_per_layer_tensor(4, 2, 2);
    auto dst = make_per_layer_tensor(4, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/3, /*chunk=*/2));
}

// Test 7: chunk_tokens > dst_seq_len must throw.
TEST(PerLayerInputsCopyTest, ChunkExceedsDstSeqLenThrows) {
    auto src = make_per_layer_tensor(8, 2, 2);
    auto dst = make_per_layer_tensor(2, 2, 2);
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/0, /*chunk=*/4));
}

// Test 8: src and dst have different per-token byte sizes (different num_layers) must throw.
TEST(PerLayerInputsCopyTest, PerTokenByteMismatchThrows) {
    auto src = make_per_layer_tensor(4, /*num_layers=*/2, 2);
    auto dst = make_per_layer_tensor(4, /*num_layers=*/3, 2);  // different num_layers
    EXPECT_ANY_THROW(ov::npuw::util::copy_per_layer_inputs_chunk_to_right(src, dst, /*offset=*/0, /*chunk=*/2));
}

// --- copy_to_right for per_layer_inputs (inlined path tests) --------------------

// Test 9: copy_to_right writes src into the right end of dst; leading bytes are left unchanged.
TEST(PerLayerInputsCopyTest, CopyToRightLeavesLeadingBytesUnchanged) {
    // src: [1, 2, 2, 2], values 0..7
    auto src = make_per_layer_tensor(2, 2, 2, 0.f);
    // dst: [1, 4, 2, 2], sequential values starting from 99 (99, 100, 101, ...)
    auto dst = make_per_layer_tensor(4, 2, 2, 99.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_to_right(src, dst));

    const auto result = to_vec(dst);
    std::vector<float> expected = {99.f,
                                   100.f,
                                   101.f,
                                   102.f,  // unchanged (token 0)
                                   103.f,
                                   104.f,
                                   105.f,
                                   106.f,  // unchanged (token 1)
                                   0.f,
                                   1.f,
                                   2.f,
                                   3.f,  // src token 0
                                   4.f,
                                   5.f,
                                   6.f,
                                   7.f};  // src token 1
    EXPECT_EQ(result, expected);
}

// Test 10: copy_to_right when src size == dst size copies everything.
TEST(PerLayerInputsCopyTest, CopyToRightSameSizeCopiesAll) {
    auto src = make_per_layer_tensor(2, 2, 2, 1.f);
    auto dst = make_per_layer_tensor(2, 2, 2, 0.f);

    ASSERT_NO_THROW(ov::npuw::util::copy_to_right(src, dst));

    EXPECT_EQ(to_vec(dst), to_vec(src));
}

}  // namespace
