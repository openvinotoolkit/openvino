// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/supported_section_type_evaluator.hpp"

#include <gtest/gtest.h>

using namespace intel_npu;

using SupportedSectionTypeEvaluatorUnitTests = ::testing::Test;

TEST_F(SupportedSectionTypeEvaluatorUnitTests, AlwaysEvaluatesToTrue) {
    ASSERT_TRUE(SupportedSectionTypeEvaluator::get_instance()->get_result());
}
