// Copyright (C) 2019 Intel Corporation
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
generateVector(size_t vec_len, uint32_t upTo = 10, uint32_t startFrom = 1, int32_t seed = 1) {
    std::vector<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type> res;

    if (seed == 1) {
        seed = static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(startFrom, upTo);

    for (int i = 0; i < vec_len; i++) {
        res.push_back(
                static_cast<typename ngraph::helpers::nGraphTypesTrait<dType>::value_type>(dist(gen)));
    }
    return res;
}

std::vector<ngraph::float16> inline generateF16Vector(size_t vec_len, uint32_t upTo = 10, uint32_t startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::float16> res;

    if (seed == 1) {
        seed = static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }

    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(startFrom, upTo);

    for (int i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::float16(static_cast<float>(dist(gen))));
    }
    return res;
}

std::vector<ngraph::bfloat16> inline generateBF16Vector(size_t vec_len, uint32_t upTo = 10, uint32_t startFrom = 1, int32_t seed = 1) {
    std::vector<ngraph::bfloat16> res;

    if (seed == 1) {
        seed = static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    std::mt19937 gen(seed);
    // chose values between this range to avoid type overrun (e.g. in case of I8 precision)
    std::uniform_int_distribution<unsigned long> dist(startFrom, upTo);

    for (int i = 0; i < vec_len; i++) {
        res.emplace_back(ngraph::bfloat16(static_cast<float>(dist(gen))));
    }
    return res;
}

template<typename fromType, typename toType>
std::vector<toType> castVector(const std::vector<fromType> &vec) {
    std::vector<toType> resVec;
    resVec.reserve(vec.size());
    for (auto& el : vec) {
        resVec.push_back(static_cast<toType>(el));
    }
    return resVec;
}

}  // namespace Utils
}  // namespace NGraphFunctions
