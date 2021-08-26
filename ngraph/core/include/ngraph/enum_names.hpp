// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/check.hpp"
#include "openvino/core/enum_names.hpp"

namespace ngraph {
using ov::EnumNames;
/// Returns the enum value matching the string
template <typename Type, typename Value>
typename std::enable_if<std::is_convertible<Value, std::string>::value, Type>::type as_enum(const Value& value) {
    return ov::as_enum<Type>(value);
}

/// Returns the string matching the enum value
template <typename Value>
const std::string& as_string(Value value) {
    return ov::as_string(value);
}
}  // namespace ngraph
