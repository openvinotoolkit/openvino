// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/progress_reporter.hpp"

#include <gtest/gtest.h>

#include "openvino/frontend/exception.hpp"

using namespace ov::frontend;

TEST(ProgressReporter_Callables, LambdaReporter) {
    const auto lambda = [](float progress, unsigned int total, unsigned int completed) {
        EXPECT_NEAR(progress, 0.5, 0.0001);
        EXPECT_EQ(total, 100);
        EXPECT_EQ(completed, 50);
    };

    ProgressReporterExtension ext{lambda};
    ext.report_progress(0.5, 100, 50);
}

TEST(ProgressReporter_Callables, RvalueLambdaReporter) {
    ProgressReporterExtension ext{[](float progress, unsigned int total, unsigned int completed) {
        EXPECT_NEAR(progress, 0.5, 0.0001);
        EXPECT_EQ(total, 100);
        EXPECT_EQ(completed, 50);
    }};

    ext.report_progress(0.5, 100, 50);
}

TEST(ProgressReporter_Callables, StructReporter) {
    struct ProgressConsumer {
        void operator()(float progress, unsigned int total, unsigned int completed) {
            EXPECT_NEAR(progress, 0.5675, 0.0001);
            EXPECT_EQ(total, 37);
            EXPECT_EQ(completed, 21);
        }
    };

    ProgressConsumer consumer;

    ProgressReporterExtension ext{consumer};
    ext.report_progress(0.5675f, 37, 21);
}

namespace {
void function_reporter(float progress, unsigned int total, unsigned int completed) {
    EXPECT_NEAR(progress, 0.2574, 0.0001);
    EXPECT_EQ(total, 101);
    EXPECT_EQ(completed, 26);
}

void reporter_stub(float, unsigned int, unsigned int) {}
}  // namespace

TEST(ProgressReporter_Callables, FunctionReporter) {
    ProgressReporterExtension ext{function_reporter};
    ext.report_progress(0.2574f, 101, 26);
}

TEST(ProgressReporter, ReportMoreStepsThanTotal) {
    ProgressReporterExtension ext{reporter_stub};
    EXPECT_THROW(ext.report_progress(0.0, 100, 101), ov::frontend::GeneralFailure);
}

TEST(ProgressReporter, ReportMoreThan100Percent) {
    ProgressReporterExtension ext{reporter_stub};
    EXPECT_THROW(ext.report_progress(1.00001f, 100, 50), ov::frontend::GeneralFailure);
}

TEST(ProgressReporter, ReportLessThanZeroPercent) {
    ProgressReporterExtension ext{reporter_stub};
    EXPECT_THROW(ext.report_progress(-100.0, 100, 50), ov::frontend::GeneralFailure);
}
