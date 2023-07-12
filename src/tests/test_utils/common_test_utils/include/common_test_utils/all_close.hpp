// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/pass/manager.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"


namespace ov {
namespace test {
namespace utils {

/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, ::testing::AssertionResult>::type all_close(
    const std::vector<T>& a,
    const std::vector<T>& b,
    T rtol = static_cast<T>(1e-5),
    T atol = static_cast<T>(1e-8)) {
    bool rc = true;
    ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
    if (a.size() != b.size()) {
        throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
    }
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i]) || !std::isfinite(a[i]) || !std::isfinite(b[i])) {
            if (count < 5) {
                ar_fail << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << a[i]
                        << " is not close to " << b[i] << " at index " << i << std::endl;
            }
            count++;
            rc = false;
        }
    }
    ar_fail << "diff count: " << count << " out of " << a.size() << std::endl;
    return rc ? ::testing::AssertionSuccess() : ar_fail;
}

/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// \returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
template <typename T>
typename std::enable_if<std::is_integral<T>::value, ::testing::AssertionResult>::type all_close(
    const std::vector<T>& a,
    const std::vector<T>& b,
    T rtol = static_cast<T>(1e-5),
    T atol = static_cast<T>(1e-8)) {
    bool rc = true;
    ::testing::AssertionResult ar_fail = ::testing::AssertionFailure();
    if (a.size() != b.size()) {
        throw std::invalid_argument("all_close: Argument vectors' sizes do not match");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        T abs_diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
        if (abs_diff > atol + rtol * b[i]) {
            // use unary + operator to force integral values to be displayed as numbers
            ar_fail << +a[i] << " is not close to " << +b[i] << " at index " << i << std::endl;
            rc = false;
        }
    }
    return rc ? ::testing::AssertionSuccess() : ar_fail;
}

/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
template <typename T>
::testing::AssertionResult all_close(const ov::Tensor& a, const ov::Tensor& b, T rtol = 1e-5f, T atol = 1e-8f) {

    // all_close(
    // const std::vector<T>& a,
    // const std::vector<T>& b,
    // T rtol = static_cast<T>(1e-5),
    // T atol = static_cast<T>(1e-8))

    std::vector<T> a_v(a.get_size());
    ov::Tensor a_t(a.get_element_type(), a.get_shape(), static_cast<void *>(a_v.data()));
    a.copy_to(a_t);

    std::vector<T> b_v(b.get_size());
    ov::Tensor b_t(b.get_element_type(), b.get_shape(), static_cast<void *>(b_v.data()));
    b.copy_to(b_t);

    return all_close(a_v, b_v, rtol, atol);
}

/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
::testing::AssertionResult all_close(const ov::Tensor& a, const ov::Tensor& b, float rtol = 1e-5f, float atol = 1e-8f);
}  // namespace utils
}  // namespace test
}  // namespace ov
