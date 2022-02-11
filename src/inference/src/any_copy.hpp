// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_parameter.hpp>
#include <istream>
#include <map>
#include <openvino/core/any.hpp>
#include <openvino/runtime/common.hpp>
#include <openvino/runtime/properties.hpp>
#include <ostream>
#include <string>

namespace ov {
std::map<std::string, std::string> any_copy(const ov::AnyMap& config_map);
}  // namespace ov
