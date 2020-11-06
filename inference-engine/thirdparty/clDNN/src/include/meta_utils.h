/*
// Copyright (c) 2017-2020 Intel Corporation
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

#include "cldnn/runtime/meta_utils.hpp"

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
