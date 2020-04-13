// Copyright (C) 2018-2020 Intel Corporation
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
generateVector(size_t vec_len) {
    std::vector<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type> res;

    std::mt19937 gen(
            static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(1, 10);

    for (int i = 0; i < vec_len; i++) {
        res.push_back(
                static_cast<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>(dist(gen)));
    }
    return res;
}

std::vector<ngraph::float16> inline generateF16Vector(size_t vec_len) {
    std::vector<ngraph::float16> res;

    std::mt19937 gen(
            static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(1, 10);

    for (int i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::float16(static_cast<float>(dist(gen))));
    }
    return res;
}

std::vector<ngraph::bfloat16> inline generateBF16Vector(size_t vec_len) {
    std::vector<ngraph::bfloat16> res;

    std::mt19937 gen(
            static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(1, 10);

    for (int i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::bfloat16(static_cast<float>(dist(gen))));
    }
    return res;
}

template<typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType> &vec) {
    std::vector<toType> resVec;
    for (auto el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions