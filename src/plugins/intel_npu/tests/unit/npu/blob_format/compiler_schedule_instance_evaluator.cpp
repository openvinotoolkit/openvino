// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_schedule_instance_evaluator.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"

using namespace intel_npu;

using CompilerScheduleInstanceEvaluatorTest = ::testing::Test;

TEST(CompilerScheduleInstanceEvaluatorTest, NullArgs) {
    OV_EXPECT_THROW(CompilerScheduleInstanceEvaluator::get_instance(ov::SoPtr<IEngineBackend>(), nullptr),
                    ov::Exception,
                    testing::_);
}
