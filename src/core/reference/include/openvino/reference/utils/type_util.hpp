// Copyright (C) 2018-2024 Intel Corporation
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
    return std::is_floating_point<U>::value || std::is_same<float16, U>::value || std::is_same<bfloat16, U>::value ||
           std::is_same<float8_e4m3, U>::value || std::is_same<float8_e5m2, U>::value ||
           std::is_same<float4_e2m1, U>::value || std::is_same<float8_e8m0, U>::value;
}
}  // namespace ov
