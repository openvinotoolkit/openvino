// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "ngraph/deprecated.hpp"

namespace ngraph {

// If operation has type_info, use it to add operation to opset
template <class T>
class HasTypeInfoMember {
    template <typename U>
    static std::false_type check(...);
    NGRAPH_SUPPRESS_DEPRECATED_START
    template <typename U>
    static auto check(int) -> decltype(std::declval<U>().type_info, std::true_type{});
    using type = decltype(check<T>(0));
    NGRAPH_SUPPRESS_DEPRECATED_END
public:
    static constexpr bool value = type::value;
};

}  // namespace ngraph
