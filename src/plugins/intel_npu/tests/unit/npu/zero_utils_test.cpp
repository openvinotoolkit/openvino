// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_utils.hpp"

#include <gtest/gtest.h>

namespace {

TEST(ZeroUtils, isSameStorageSpellingAcceptsI2AndU2InBothOrders) {
    EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u2, ov::element::i2));
    EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::i2, ov::element::u2));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u4, ov::element::i4));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u2, ov::element::u4));
}

}  // namespace
