// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/supported_section_type_evaluator.hpp"

#include <gtest/gtest.h>

using namespace intel_npu;

using SupportedSectionTypeEvaluatorTest = ::testing::Test;

TEST_F(SupportedSectionTypeEvaluatorTest, AlwaysEvaluatesToTrue) {
    ASSERT_TRUE(SupportedSectionTypeEvaluator::get_instance()->get_result());
}
