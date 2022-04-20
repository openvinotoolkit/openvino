// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "log.hpp"

namespace MultiDevicePlugin {
uint32_t Log::defaultLogLevel = static_cast<uint32_t>(LogLevel::LOG_NONE);
std::vector<std::string> Log::validFormat = {"u", "d", "s", "ld", "lu", "lf"};
} // namespace MultiDevicePlugin
