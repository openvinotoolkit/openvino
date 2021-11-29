// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <random>
#include <chrono>

#include <ngraph/type/element_type.hpp>
#include "ngraph_helpers.hpp"

namespace NGraphFunctions {
namespace Utils {


template<ngraph::element::Type_t dType>
std::vector<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type> inline
generateVector(size_t vec_len,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type upTo = 10,
               typename ngraph::helpers::nGraphTypesTrait<dType>::value_type startFrom = 1,
               int32_t seed = 1) {
    using dataType = typename ngraph::helpers::nGraphTypesTrait<dType>::value_type;
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

std::vector<ngraph::float16> inline
generateF16Vector(size_t vec_len, ngraph::float16 upTo = 10, ngraph::float16 startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::float16> res(vec_len);
    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ngraph::float16(dist(gen));
    }
    return res;
}

std::vector<ngraph::bfloat16> inline
generateBF16Vector(size_t vec_len, ngraph::bfloat16 upTo = 10, ngraph::bfloat16 startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::bfloat16> res(vec_len);

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_real_distribution<float> dist(startFrom, upTo);;
    // explicitly include data range borders to avoid missing the corner values while data generation
    res[0] = startFrom;
    res[vec_len - 1] = upTo;
    for (size_t i = 1; i < vec_len - 1; i++) {
        res[i] = ngraph::bfloat16(dist(gen));
    }
    return res;
}

template<typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType> &vec) {
    std::vector<toType> resVec;
    resVec.reserve(vec.size());
    for (auto &el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions
