// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include <random>
#include <type_traits>
#include <vector>

#include "ngraph_helpers.hpp"

namespace NGraphFunctions {
namespace Utils {

// std::uniform_int_distribution can't work with char/int8/uchar/uint8 types
template <typename T>
typename std::enable_if<sizeof(T) == 1, unsigned short>::type getIntegerDistributionType();
template <typename T>
typename std::enable_if<sizeof(T) != 1, T>::type getIntegerDistributionType();

// integer version
template <ngraph::element::Type_t dType,
          typename dataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>
std::vector<
    typename std::enable_if<!std::is_same<dataType, bool>::value && std::is_integral<dataType>::value, dataType>::type>
generateVector(size_t vec_len,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1,
               int32_t seed = 1) {
    std::vector<dataType> res(vec_len);
    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<decltype(getIntegerDistributionType<dataType>())> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = dataType(dist(gen));
    }

    return res;
}

// floating point version
template <ngraph::element::Type_t dType,
          typename dataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>
std::vector<typename std::enable_if<std::is_floating_point<dataType>::value, dataType>::type> generateVector(
    size_t vec_len,
    typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
    typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1,
    int32_t seed = 1) {
    std::vector<dataType> res(vec_len);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<dataType> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = dist(gen);
    }

    return res;
}

// boolean version
template <ngraph::element::Type_t dType,
          typename dataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>
std::vector<typename std::enable_if<std::is_same<dataType, bool>::value, bool>::type> generateVector(
    size_t vec_len,
    typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
    typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1,
    int32_t seed = 1) {
    std::vector<dataType> res(vec_len);
    std::mt19937 gen(seed);
    std::bernoulli_distribution dist;
    for (size_t i = 0; i < vec_len; i++) {
        res[i] = dist(gen);
    }
    return res;
}

// float16 and bfloat16 version
template <ngraph::element::Type_t dType,
          typename dataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>
std::vector<typename std::enable_if<std::is_same<dataType, ngraph::float16>::value ||
                                        std::is_same<dataType, ngraph::bfloat16>::value,
                                    dataType>::type>
generateVector(size_t vec_len,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1,
               int32_t seed = 1) {
    std::vector<dataType> res(vec_len);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = dataType(dist(gen));
    }
    return res;
}

template <typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType>& vec) {
    std::vector<toType> resVec;
    resVec.reserve(vec.size());
    for (const auto& el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions
