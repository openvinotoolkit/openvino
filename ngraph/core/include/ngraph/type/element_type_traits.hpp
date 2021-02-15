//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    template <element::Type_t>
    struct element_type_traits
    {
    };

    template <>
    struct element_type_traits<element::Type_t::boolean>
    {
        using value_type = char;
    };

    template <>
    struct element_type_traits<element::Type_t::bf16>
    {
        using value_type = bfloat16;
    };

    template <>
    struct element_type_traits<element::Type_t::f16>
    {
        using value_type = float16;
    };

    template <>
    struct element_type_traits<element::Type_t::f32>
    {
        using value_type = float;
    };

    template <>
    struct element_type_traits<element::Type_t::f64>
    {
        using value_type = double;
    };

    template <>
    struct element_type_traits<element::Type_t::i8>
    {
        using value_type = int8_t;
    };

    template <>
    struct element_type_traits<element::Type_t::i16>
    {
        using value_type = int16_t;
    };

    template <>
    struct element_type_traits<element::Type_t::i32>
    {
        using value_type = int32_t;
    };

    template <>
    struct element_type_traits<element::Type_t::i64>
    {
        using value_type = int64_t;
    };

    template <>
    struct element_type_traits<element::Type_t::u1>
    {
        using value_type = int8_t;
    };

    template <>
    struct element_type_traits<element::Type_t::u8>
    {
        using value_type = uint8_t;
    };

    template <>
    struct element_type_traits<element::Type_t::u16>
    {
        using value_type = uint16_t;
    };

    template <>
    struct element_type_traits<element::Type_t::u32>
    {
        using value_type = uint32_t;
    };

    template <>
    struct element_type_traits<element::Type_t::u64>
    {
        using value_type = uint64_t;
    };
}
