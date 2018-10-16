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

namespace cldnn
{

struct primitive;

namespace meta
{

template <class... T>
struct pack {};

template <class T, class... U>
constexpr bool is_any_of_v = is_any_of<T, U...>::value;

//helper type for deducing return type from member function pointer
//doesn't require passing arguments like std::result_of
template <class T>
struct deduce_ret_type;

template <class Ret, class C, class... Args>
struct deduce_ret_type<Ret(C::*)(Args...)>
{
    using type = Ret;
};

template <class T>
using deduce_ret_type_t = typename deduce_ret_type<T>::type;

template <class T>
constexpr bool always_false_v = always_false<T>::value;

template <class T>
struct is_primitive : public std::integral_constant<bool,
                                                    std::is_base_of<primitive, T>::value &&
                                                    !std::is_same<primitive, std::remove_cv_t<T>>::value &&
                                                    std::is_same<T, std::remove_cv_t<T>>::value> {};

template <class T>
constexpr bool is_primitive_v = is_primitive<T>::value;

template <class T>
struct is_api_primitive : public std::integral_constant<bool,
                                                    is_primitive_v<T> &&
                                                    !std::is_base_of<internal_primitive, T>::value> {};

template <class T>
constexpr bool is_api_primitive_v = is_api_primitive<T>::value;

template <class T>
struct is_internal_primitive : public std::integral_constant<bool,
                                                    std::is_base_of<internal_primitive, T>::value &&
                                                    !std::is_same<internal_primitive, std::remove_cv_t<T>>::value &&
                                                    std::is_same<T, std::remove_cv_t<T>>::value> {};

template <class T>
constexpr bool is_internal_primitive_v = is_internal_primitive<T>::value;

}
}