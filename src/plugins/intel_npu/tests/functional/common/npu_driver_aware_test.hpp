// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

inline ov::test::utils::DriverType getDriverType() {
    return ov::test::utils::NpuTestEnvConfig::getInstance().driver_type;
}

// Returns true when the driver type matches `type`.
inline bool isDriverType(ov::test::utils::DriverType type) {
    return getDriverType() == type;
}

}  // namespace ov::test::behavior
