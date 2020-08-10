// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <gtest/gtest.h>
#include <ngraph_functions/pass/convert_prc.hpp>
#include "ng_test_utils.hpp"

namespace FuncTestUtils {
template<class T>
void ComparableNGTestCommon::compareTypedVectors(const std::vector<std::vector<T>>& v1, const std::vector<std::vector<T>>& v2) {
    InferenceEngine::Precision precision;
    ASSERT_EQ(v1.size(), v2.size());
    auto size = v1.size();
    for (std::size_t idx = 0; idx < size; ++idx) {
        const auto& expected = v1[idx];
        const auto& actual = v2[idx];
        if (std::is_same<T, int>::value) {
            precision = InferenceEngine::Precision::I32;
        } else if (std::is_same<T, float>::value) {
            precision = InferenceEngine::Precision::FP32;
        } else {
            THROW_IE_EXCEPTION << "Precision not supported";
        }

        compareValues(v1[idx].data(), v2[idx].data(), v1.size(), precision);
    }
}
void ComparableNGTestCommon::compareBytes(const std::vector<std::vector<std::uint8_t>>& expectedVector,
                                          const std::vector<std::vector<std::uint8_t>>& actualVector,
                                          const InferenceEngine::Precision precision) {
    for (std::size_t idx = 0; idx < expectedVector.size(); ++idx) {
        const auto& expected = expectedVector[idx];
        const auto& actual = actualVector[idx];
        ASSERT_EQ(expectedVector.size(), actualVector.size());
        const unsigned char *expectedBuffer = expected.data();
        const unsigned char *actualBuffer = actual.data();
        auto size = actual.size();
        compareValues(expectedBuffer, actualBuffer, size, precision);
    }
}
template<class T>
void ComparableNGTestCommon::compareValues(const T *expected, const T *actual, std::size_t size, T thr) {
    std::cout << std::endl;
    for (std::size_t i = 0; i < size; ++i) {
        const auto &ref = expected[i];
        const auto &res = actual[i];
        const auto absoluteDifference = std::abs(res - ref);
        if (absoluteDifference <= thr) {
            continue;
        }

        const auto max = std::max(std::abs(res), std::abs(ref));
        ASSERT_TRUE(max != 0 && ((absoluteDifference / max) <= thr))
                                    << "Relative comparison of values expected: " << ref << " and actual: " << res
                                    << " at index " << i << " with t " << thr
                                    << " failed";
    }
}

    void ComparableNGTestCommon::compareValues(const void *expected, const void *actual, std::size_t size, const InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            FuncTestUtils::ComparableNGTestCommon::compareValues(
                    reinterpret_cast<const float *>(expected), reinterpret_cast<const float *>(actual),
                    size, threshold);
            break;
        case InferenceEngine::Precision::I32:
            FuncTestUtils::ComparableNGTestCommon::compareValues(
                    reinterpret_cast<const std::int32_t *>(expected),
                    reinterpret_cast<const std::int32_t *>(actual), size, 0);
            break;
        default:
            FAIL() << "Comparator for " << precision << " precision isn't supported";
    }
}
std::vector<std::vector<std::uint8_t>> ComparableNGTestCommon::CalculateRefs(std::shared_ptr<ngraph::Function> _function,
        std::vector<std::vector<std::uint8_t>> _inputs) {
    // nGraph interpreter does not support f16
    // IE converts f16 to f32
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::f16, ngraph::element::Type_t::f32>().run_on_function(_function);
    _function->validate_nodes_and_infer_types();
    return ngraph::helpers::interpreterFunction(_function, _inputs, ::ngraph::element::Type_t::undefined);
}
}  // namespace FuncTestUtils