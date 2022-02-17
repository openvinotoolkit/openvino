// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <memory>
#include <string>
#include <type_traits>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

float half_to_float(uint16_t value);
uint16_t float_to_half(float value);

// There is no portable half precision floating point support.
// Using wrapped integral type with the same size and alignment restrictions.
class half_impl {
public:
    half_impl() = default;

    template <typename T, typename = typename std::enable_if<!std::is_floating_point<T>::value>::type>
    explicit half_impl(T data, int /*direct_creation_tag*/) : _data(data) {}

    operator uint16_t() const { return _data; }
    operator float() const {
        return half_to_float(_data);
    }

    explicit half_impl(float value)
        : _data(float_to_half(value))
    {}

    template <typename T, typename = typename std::enable_if<std::is_convertible<T, float>::value>::type>
    explicit half_impl(T value)
        : half_impl(static_cast<float>(value))
    {}

private:
    uint16_t _data;
};

// Use complete implementation if necessary.
#if defined HALF_HALF_HPP
using half_t = half;
#else
using half_t = half_impl;
#endif

}  // namespace cldnn
