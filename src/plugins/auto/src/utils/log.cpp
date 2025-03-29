// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "log.hpp"
namespace ov {
namespace auto_plugin {
ov::log::Level Log::default_log_level = ov::log::Level::NO;
std::vector<std::string> Log::valid_format = {"u", "d", "s", "ld", "lu", "lf"};
} // namespace auto_plugin
} // namespace ov
