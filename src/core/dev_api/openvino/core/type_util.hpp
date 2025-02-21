// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"

namespace ov {

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to any of the specified types
template <typename Type, typename... Types, typename Value>
bool is_type_any_of(Value value) {
    return is_type<Type>(value) || (is_type_any_of<Types>(value) || ...);
}

}  // namespace ov
