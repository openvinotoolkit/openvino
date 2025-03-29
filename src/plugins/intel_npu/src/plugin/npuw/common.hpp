// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {

using DeviceProperties = std::unordered_map<std::string, ov::AnyMap>;

}  // namespace npuw
}  // namespace ov
