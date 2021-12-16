// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include <common/progress_reporter_extension.hpp>

using namespace ov::frontend;

TEST(ProgressReporter_ProgressCounter, BasicAdvance) {
    ProgressCounter progress{10};
    const auto percent = progress.advance();
    EXPECT_EQ(progress.completed_steps(), 1);
    EXPECT_NEAR(percent, 0.1, 0.01);

    progress++;
    EXPECT_EQ(progress.completed_steps(), 2);
    EXPECT_NEAR(progress.current_progress(), 0.2, 0.01);

    ++progress;
    EXPECT_EQ(progress.completed_steps(), 3);
    EXPECT_NEAR(progress.current_progress(), 0.3, 0.01);

    ++progress++;
    EXPECT_EQ(progress.completed_steps(), 5);
    EXPECT_NEAR(progress.current_progress(), 0.5, 0.01);
}

TEST(ProgressReporter_ProgressCounter, PostCompleteAdvance) {
    ProgressCounter progress{2};
    const auto percent = progress.advance(3);
    EXPECT_EQ(progress.completed_steps(), 2);
    EXPECT_NEAR(percent, 1.0, 0.01);
}
