// Copyright (C) 2018-2022 Intel Corporation
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

namespace ngraph {
namespace test {
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
::testing::AssertionResult all_close(const std::shared_ptr<ngraph::runtime::Tensor>& a,
                                     const std::shared_ptr<ngraph::runtime::Tensor>& b,
                                     T rtol = 1e-5f,
                                     T atol = 1e-8f) {
    if (a->get_shape() != b->get_shape()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different shapes";
    }

    return all_close(read_vector<T>(a), read_vector<T>(b), rtol, atol);
}

/// \brief Same as numpy.allclose
/// \param as First tensors to compare
/// \param bs Second tensors to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
template <typename T>
::testing::AssertionResult all_close(const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& as,
                                     const std::vector<std::shared_ptr<ngraph::runtime::Tensor>>& bs,
                                     T rtol,
                                     T atol) {
    if (as.size() != bs.size()) {
        return ::testing::AssertionFailure() << "Cannot compare tensors with different sizes";
    }
    for (size_t i = 0; i < as.size(); ++i) {
        auto ar = all_close(as[i], bs[i], rtol, atol);
        if (!ar) {
            return ar;
        }
    }
    return ::testing::AssertionSuccess();
}
}  // namespace test
}  // namespace ngraph

namespace ov {
namespace test {
/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
template <typename T>
::testing::AssertionResult all_close(const ov::Tensor& a, const ov::Tensor& b, T rtol = 1e-5f, T atol = 1e-8f) {
    auto a_t = std::make_shared<ngraph::runtime::HostTensor>(a.get_element_type(), a.get_shape(), a.data());
    auto b_t = std::make_shared<ngraph::runtime::HostTensor>(b.get_element_type(), b.get_shape(), b.data());

    return ngraph::test::all_close(a_t, b_t, rtol, atol);
}

/// \brief Same as numpy.allclose
/// \param a First tensor to compare
/// \param b Second tensor to compare
/// \param rtol Relative tolerance
/// \param atol Absolute tolerance
/// Returns true if shapes match and for all elements, |a_i-b_i| <= atol + rtol*|b_i|.
::testing::AssertionResult all_close(const ov::Tensor& a, const ov::Tensor& b, float rtol = 1e-5f, float atol = 1e-8f);
}  // namespace test
}  // namespace ov
