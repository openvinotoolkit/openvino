// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "openvino/core/type/element_type_traits.hpp"

namespace NGraphFunctions {
namespace Utils {

template <ov::element::Type_t dType>
std::vector<typename ov::element_type_traits<dType>::value_type> inline generateVector(
    size_t vec_len,
    typename ov::element_type_traits<dType>::value_type upTo = 10,
    typename ov::element_type_traits<dType>::value_type startFrom = 1,
    int32_t seed = 1) {
    using dataType = typename ov::element_type_traits<dType>::value_type;
    std::vector<dataType> res(vec_len);

    std::mt19937 gen(seed);
    if (std::is_floating_point<dataType>()) {
        // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
        std::uniform_real_distribution<double> dist(static_cast<double>(startFrom), static_cast<double>(upTo));
        // explicitly include data range borders to avoid missing the corner values while data generation
        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    } else if (std::is_same<bool, dataType>()) {
        std::bernoulli_distribution dist;
        for (size_t i = 0; i < vec_len; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    } else {
        // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
        std::uniform_int_distribution<long> dist(static_cast<long>(startFrom), static_cast<long>(upTo));
        // explicitly include data range borders to avoid missing the corner values while data generation
        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            res[i] = static_cast<dataType>(dist(gen));
        }
        return res;
    }
}

template <>
std::vector<ov::float16> inline generateVector<ov::element::Type_t::f16>(size_t vec_len,
                                                                         ov::float16 upTo,
                                                                         ov::float16 startFrom,
                                                                         int32_t seed) {
    std::vector<ov::float16> res(vec_len);
    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ov::float16(dist(gen));
    }
    return res;
}

template <>
std::vector<ov::bfloat16> inline generateVector<ov::element::Type_t::bf16>(size_t vec_len,
                                                                           ov::bfloat16 upTo,
                                                                           ov::bfloat16 startFrom,
                                                                           int32_t seed) {
    std::vector<ov::bfloat16> res(vec_len);

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    ;
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ov::bfloat16(dist(gen));
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
