// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "log.hpp"
namespace ov {
namespace auto_plugin {
uint32_t Log::default_log_level = static_cast<uint32_t>(LogLevel::LOG_NONE);
std::vector<std::string> Log::valid_format = {"u", "d", "s", "ld", "lu", "lf"};
} // namespace auto_plugin
} // namespace ov
