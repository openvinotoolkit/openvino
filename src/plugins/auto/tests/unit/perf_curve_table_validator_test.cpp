// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <tuple>

#include "plugin_config.hpp"

using ov::auto_plugin::PerfCurveTableValidator;
using ov::intel_auto::PerfCurveTable;

// (value, expected is_valid) — PerfCurveTableValidator is the single source of truth for the
// semantic rules (device whitelist, non-empty curve, utilization range, finite non-negative score).
using PerfCurveValidatorParams = std::tuple<ov::Any, bool>;

class PerfCurveTableValidatorTest : public ::testing::TestWithParam<PerfCurveValidatorParams> {};

TEST_P(PerfCurveTableValidatorTest, is_valid) {
    const auto& [value, expected] = GetParam();
    PerfCurveTableValidator validator;
    EXPECT_EQ(validator.is_valid(value), expected);
}

const std::vector<PerfCurveValidatorParams> perfCurveValidatorConfigs = {
    // Empty table disables the feature and is valid (matches the registered default).
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{}), true},
    // Whitelisted devices, in-range utilization, finite non-negative scores.
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{0, 0.f}, {100, 100.f}}}}), true},
    PerfCurveValidatorParams{
        ov::Any(PerfCurveTable{{"iGPU", {{50, 25.5f}}}, {"dGPU", {{0, 0.f}}}, {"NPU", {{100, 40.f}}}}),
        true},
    // Whitelist violation: unknown device and the base "GPU" (must be iGPU/dGPU).
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"XXX", {{0, 0.f}}}}), false},
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"GPU", {{0, 0.f}}}}), false},
    // Empty curve for a device.
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {}}}), false},
    // Utilization key out of [0, 100].
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{101, 10.f}}}}), false},
    // Negative score.
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{0, -1.f}}}}), false},
    // Non-finite scores.
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{0, std::numeric_limits<float>::quiet_NaN()}}}}), false},
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{0, std::numeric_limits<float>::infinity()}}}}), false},
    // Wrong Any type that cannot be parsed into a PerfCurveTable.
    PerfCurveValidatorParams{ov::Any(std::string("not-a-table")), false},
    PerfCurveValidatorParams{ov::Any(42), false},
    // Validate the minimum and maximum utilization values with corresponding scores.
    PerfCurveValidatorParams{ov::Any(PerfCurveTable{{"CPU", {{0, 0.f}, {100, 100.f}}}}), true},
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_PerfCurveTableValidator,
                         PerfCurveTableValidatorTest,
                         ::testing::ValuesIn(perfCurveValidatorConfigs));
