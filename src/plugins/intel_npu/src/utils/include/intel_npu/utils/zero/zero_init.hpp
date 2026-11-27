// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// The level zero driver, device and context used to be set up in this one header. They are in their own headers now,
// this one only keeps the includes that already point here working.
#include <map>
#include <optional>
#include <string>
#include <string_view>

#include "intel_npu/utils/zero/zero_context.hpp"
#include "intel_npu/utils/zero/zero_device.hpp"
#include "intel_npu/utils/zero/zero_driver.hpp"
