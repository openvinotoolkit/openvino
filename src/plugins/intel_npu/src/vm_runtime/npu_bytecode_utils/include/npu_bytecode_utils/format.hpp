//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace intel_npu::vm {

// Format the string by replacing "{}" placeholders in the input with the provided arguments. If there are
// more arguments than placeholders, the extra arguments are appended to the end of the message. If there are more
// placeholders than arguments, the extra placeholders are left as-is in the message
std::string formatString(std::string_view format, const std::vector<std::string>& args);

}  // namespace intel_npu::vm
