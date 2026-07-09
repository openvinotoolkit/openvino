// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "common/npu_test_env_cfg.hpp"

// Example: NPU_SKIP_IF_DRIVER_TYPE_IS(PV, "C#12345: feature X not supported on PV driver");
#define NPU_SKIP_IF_DRIVER_TYPE_IS(type, reason)                                             \
    do {                                                                                      \
        if (::ov::test::behavior::isDriverType(::ov::test::utils::DriverType::type)) {      \
            GTEST_SKIP() << (reason);                                                        \
        }                                                                                     \
    } while (0)

#define NPU_SKIP_UNLESS_DRIVER_TYPE_IS(type, reason)                                         \
    do {                                                                                      \
        if (!::ov::test::behavior::isDriverType(::ov::test::utils::DriverType::type)) {     \
            GTEST_SKIP() << (reason);                                                        \
        }                                                                                     \
    } while (0)

namespace ov::test::behavior {

// Returns the driver type set via CLI --driver_type argument in main().
inline std::optional<ov::test::utils::DriverType> getDriverType() {
    return ov::test::utils::g_driver_type;
}

// Returns true when the driver type matches `type`.
inline bool isDriverType(ov::test::utils::DriverType type) {
    const auto dt = getDriverType();
    return dt.has_value() && *dt == type;
}

}  // namespace ov::test::behavior
