// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "log.hpp"

namespace MultiDevicePlugin {
uint32_t Log::defaultLogLevel = static_cast<uint32_t>(LogLevel::INFO) |
                                static_cast<uint32_t>(LogLevel::WARN) |
                                static_cast<uint32_t>(LogLevel::ERROR) |
                                static_cast<uint32_t>(LogLevel::FATAL);
} // namespace MultiDevicePlugin
