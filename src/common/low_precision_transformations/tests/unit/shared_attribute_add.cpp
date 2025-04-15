// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"

using LPT_ReshapeTransformation = ::testing::Test;

TEST(LPT_SharedAttribute, assign) {
    const auto attribute1 = ov::PrecisionPreservedAttribute();
    ASSERT_EQ(1ul, attribute1.attribute->sharedValue->getAttributes().size());

    const auto attribute2 = ov::AvgPoolPrecisionPreservedAttribute();
    ASSERT_EQ(1ul, attribute2.attribute->sharedValue->getAttributes().size());

    ov::pass::low_precision::NetworkHelper::reassign<ov::AvgPoolPrecisionPreservedAttribute>(
        attribute1.attribute->sharedValue,
        { attribute1.attribute, attribute2.attribute });

    ASSERT_EQ(2ul, attribute1.attribute->sharedValue->getAttributes().size());
    ASSERT_EQ(2ul, attribute2.attribute->sharedValue->getAttributes().size());
}
