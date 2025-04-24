// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "low_precision/network_helper.hpp"

using LPT_ReshapeTransformation = ::testing::Test;

TEST(LPT_UpdateReshapeValuesTransformation, updateReshapeValues_3_3_32_1_to_1_1_32_1) {
    ASSERT_EQ(
        ov::Shape({1, 1, 32, 1}),
        ov::pass::low_precision::NetworkHelper::updateReshapeValues({ 1, 32 }, { 9, 32 }, { 3, 3, 32, 1 }));
}
