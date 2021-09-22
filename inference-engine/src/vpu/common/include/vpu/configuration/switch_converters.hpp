// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

namespace vpu {

const std::unordered_map<std::string, bool>& string2switch();
const std::unordered_map<bool, std::string>& switch2string();

}  // namespace vpu
