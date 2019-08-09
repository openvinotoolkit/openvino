/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include <type_traits>
#include "api/CPP/meta_utils.hpp"
#include "internal_primitive.h"

namespace cldnn {

struct primitive;

namespace meta {

template <class... T>
struct pack {};

// helper type for deducing return type from member function pointer
// doesn't require passing arguments like std::result_of
template <class T>
struct deduce_ret_type;

template <class Ret, class C, class... Args>
struct deduce_ret_type<Ret (C::*)(Args...)> {
    using type = Ret;
};

template <class T>
using deduce_ret_type_t = typename deduce_ret_type<T>::type;

template <class T>
struct is_primitive
    : public std::integral_constant<bool,
                                    std::is_base_of<primitive, T>::value &&
                                        !std::is_same<primitive, typename std::remove_cv<T>::type>::value &&
                                        std::is_same<T, typename std::remove_cv<T>::type>::value> {};

template <class T>
struct is_api_primitive
    : public std::integral_constant<bool, is_primitive<T>::value && !std::is_base_of<internal_primitive, T>::value> {};

template <class T>
struct is_internal_primitive
    : public std::integral_constant<bool,
                                    std::is_base_of<internal_primitive, T>::value &&
                                        !std::is_same<internal_primitive, typename std::remove_cv<T>::type>::value &&
                                        std::is_same<T, typename std::remove_cv<T>::type>::value> {};

}  // namespace meta
}  // namespace cldnn