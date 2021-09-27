// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/utils.hpp"

#include <type_traits>

namespace cldnn {

struct primitive;

namespace meta {

template <class T>
struct is_primitive
    : public std::integral_constant<bool,
                                    std::is_base_of<primitive, T>::value &&
                                        !std::is_same<primitive, typename std::remove_cv<T>::type>::value &&
                                        std::is_same<T, typename std::remove_cv<T>::type>::value> {};


}  // namespace meta
}  // namespace cldnn
