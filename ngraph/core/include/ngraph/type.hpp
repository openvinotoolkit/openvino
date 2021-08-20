// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type.hpp"

namespace ngraph {
using ov::DiscreteTypeInfo;

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
/// Type*/shared_ptr<Type>
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<decltype(std::declval<Value>()->get_type_info().is_castable(Type::type_info)), bool>::value,
    bool>::type
is_type(Value value) {
    return ov::is_type<Type>(value);
}

/// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<std::is_convertible<decltype(static_cast<Type*>(std::declval<Value>())), Type*>::value,
                        Type*>::type
as_type(Value value) {
    return ov::as_type<Type>(value);
}

/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<decltype(std::static_pointer_cast<Type>(std::declval<Value>())), std::shared_ptr<Type>>::value,
    std::shared_ptr<Type>>::type
as_type_ptr(Value value) {
    return ov::as_type_ptr<Type>(value);
}
}  // namespace ngraph
