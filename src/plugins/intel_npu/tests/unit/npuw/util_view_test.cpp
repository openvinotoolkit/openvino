// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "util.hpp"

namespace {

ov::SoPtr<ov::ITensor> make_test_tensor(const ov::Shape& shape) {
    return ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape));
}

}  // namespace

TEST(NPUWViewTest, InBoundsWindowIsAccepted) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    auto view = ov::npuw::util::view(src, /*dim=*/2, /*offset=*/2, /*len=*/4);

    EXPECT_EQ(view->get_shape(), (ov::Shape{1, 2, 4, 4}));
}

TEST(NPUWViewTest, WindowTouchingTheEndIsAccepted) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    auto view = ov::npuw::util::view(src, /*dim=*/2, /*offset=*/4, /*len=*/4);

    EXPECT_EQ(view->get_shape(), (ov::Shape{1, 2, 4, 4}));
}

// A view is a borrowed pointer into the source tensor's buffer. A window that runs past the end of
// the sequence dimension would hand the NPU driver memory the source tensor does not own, so it
// must be rejected rather than silently produce an out-of-bounds tensor.
TEST(NPUWViewTest, WindowPastTheEndIsRejected) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    EXPECT_THROW(ov::npuw::util::view(src, /*dim=*/2, /*offset=*/6, /*len=*/4), ov::Exception);
}

TEST(NPUWViewTest, OffsetPastTheEndIsRejected) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    EXPECT_THROW(ov::npuw::util::view(src, /*dim=*/2, /*offset=*/9, /*len=*/0), ov::Exception);
}

// Callers in the HFA tile path compute the mask offset as a signed int64_t difference that can go
// negative on a malformed blob; it reaches view() as a huge std::size_t and must not wrap around
// the bounds check into a wild pointer.
TEST(NPUWViewTest, NegativeOffsetWrappedToSizeTIsRejected) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});
    const int64_t negative_offset = -4;

    EXPECT_THROW(ov::npuw::util::view(src, /*dim=*/2, static_cast<std::size_t>(negative_offset), /*len=*/4),
                 ov::Exception);
}

TEST(NPUWViewTest, HugeLengthIsRejected) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    EXPECT_THROW(ov::npuw::util::view(src, /*dim=*/2, /*offset=*/0, /*len=*/std::numeric_limits<std::size_t>::max()),
                 ov::Exception);
}

TEST(NPUWViewTest, OutOfRangeDimIsRejected) {
    auto src = make_test_tensor(ov::Shape{1, 2, 8, 4});

    EXPECT_THROW(ov::npuw::util::view(src, /*dim=*/4, /*offset=*/0, /*len=*/1), ov::Exception);
}
