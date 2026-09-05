// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>

#include "infer_request_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

class SliceableTensor {
public:
    explicit SliceableTensor(const ov::Shape& shape) : m_tensor(ov::element::f32, shape) {}

    ov::SoPtr<ov::ITensor> impl() const {
        return ov::get_tensor_impl(m_tensor);
    }

private:
    ov::Tensor m_tensor;
};

// These tests pin the bounds checks in ov::npuw::util::make_tensor_slice(), the single choke point
// every KV-slice caller funnels through.
TEST(MakeTensorSliceBoundsTest, ValidSliceProducesExpectedShape) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});

    ov::SoPtr<ov::ITensor> slice;
    ASSERT_NO_THROW(slice = ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/2u, /*end=*/5u));
    EXPECT_EQ(slice->get_shape(), (ov::Shape{1, 4, 3, 16}));
}

TEST(MakeTensorSliceBoundsTest, FullExtentSliceIsAccepted) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});

    ov::SoPtr<ov::ITensor> slice;
    ASSERT_NO_THROW(slice = ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/0u, /*end=*/8u));
    EXPECT_EQ(slice->get_shape(), (ov::Shape{1, 4, 8, 16}));
}

TEST(MakeTensorSliceBoundsTest, AxisEqualToRankIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/4u, /*start=*/0u, /*end=*/1u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, AxisFarBeyondRankIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/0xFFFFFFFFu, /*start=*/0u, /*end=*/1u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, EndBeyondExtentIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/0u, /*end=*/9u),
                 ov::Exception);
}

TEST(MakeTensorSliceBoundsTest, InvertedIntervalIsRejected) {
    SliceableTensor tensor(ov::Shape{1, 4, 8, 16});
    EXPECT_THROW(ov::npuw::util::make_tensor_slice(tensor.impl(), /*dim=*/2u, /*start=*/5u, /*end=*/2u),
                 ov::Exception);
}

}  // namespace
