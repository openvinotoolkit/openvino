// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
/**
 * @brief Check if T is OpenVINO floating point precision.
 *
 * @return True if OpenVino floating point precision.
 */
template <class T>
constexpr bool is_floating_point() {
    using U = typename std::decay<T>::type;
    return std::is_floating_point<U>::value || std::is_same<float16, U>::value || std::is_same<bfloat16, U>::value;
}

/**
 * @brief Check if ov::element::Type_t is signed binary precision.
 *
 * @tparam ET  Element type for check.
 * @return True if ET is signed binary precision otherwise false.
 */
template <element::Type_t ET>
constexpr bool is_signed_binary() {
    return (ET == element::Type_t::i4) || (ET == element::Type_t::nf4);
}

/**
 * @brief Check if ov::element::Type_t is unsigned binary precision.
 *
 * @tparam ET  Element type for check.
 * @return True if ET is unsigned binary precision otherwise false.
 */
template <element::Type_t ET>
constexpr bool is_unsigned_binary() {
    return (ET == element::Type_t::u1) || (ET == element::Type_t::u4);
}
}  // namespace ov
