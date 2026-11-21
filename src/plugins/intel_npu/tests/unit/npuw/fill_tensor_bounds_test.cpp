// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>

#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "util.hpp"

namespace {

// ov::npuw::util::fill_tensor()'s offset is frequently derived from a subtraction of two
// independently-sourced extents (e.g. a chunk size vs. a restored port size, see
// LLMInferRequest::infer_chunked_prefill()); an unchecked underflow there turns into a wild
// pointer passed to std::fill. These tests pin the bounds check added to catch that.
TEST(FillTensorBoundsTest, OffsetWithinRangeIsAccepted) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{8});
    auto impl = ov::get_tensor_impl(tensor);
    EXPECT_NO_THROW(ov::npuw::util::fill_tensor<int64_t>(impl, 0, /*offset=*/4u));
}

TEST(FillTensorBoundsTest, OffsetEqualToSizeIsAccepted) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{8});
    auto impl = ov::get_tensor_impl(tensor);
    // Fills nothing, but must not be rejected: a last (fully-sized) chunk legitimately computes
    // offset == size.
    EXPECT_NO_THROW(ov::npuw::util::fill_tensor<int64_t>(impl, 0, /*offset=*/8u));
}

TEST(FillTensorBoundsTest, OffsetBeyondSizeIsRejected) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{8});
    auto impl = ov::get_tensor_impl(tensor);
    EXPECT_THROW(ov::npuw::util::fill_tensor<int64_t>(impl, 0, /*offset=*/9u), ov::Exception);
}

// The underflow this guards against: a blob-controlled chunk size larger than the restored port
// produces `offset = port_size - chunk_size`, which wraps to a huge size_t.
TEST(FillTensorBoundsTest, UnderflowedOffsetIsRejected) {
    ov::Tensor tensor(ov::element::i64, ov::Shape{8});
    auto impl = ov::get_tensor_impl(tensor);
    const size_t underflowed_offset = static_cast<size_t>(8u) - static_cast<size_t>(1024u);
    EXPECT_THROW(ov::npuw::util::fill_tensor<int64_t>(impl, 0, underflowed_offset), ov::Exception);
}

}  // namespace
