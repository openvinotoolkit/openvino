// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "common/npu_test_env_cfg.hpp"

namespace ov::test::behavior {

// Driver type query functions.
// Use these in test bodies or conditionally branching test logic to check which driver
// is currently under test (PV / RELEASE / LATEST), as set via CLI --driver_type argument.

// Returns the driver type set via CLI --driver_type argument in main().
// Empty optional when running on a developer machine (CLI arg not provided).
inline std::optional<ov::test::utils::DriverType> getDriverType() {
    return ov::test::utils::g_driver_type;
}

// Returns true when the driver type matches `type`.
// Always returns false when the CLI arg is not set (no unintended skips on dev machines).
inline bool isDriverType(ov::test::utils::DriverType type) {
    const auto dt = getDriverType();
    return dt.has_value() && *dt == type;
}

}  // namespace ov::test::behavior
