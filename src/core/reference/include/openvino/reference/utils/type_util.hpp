// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
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
}  // namespace ov
