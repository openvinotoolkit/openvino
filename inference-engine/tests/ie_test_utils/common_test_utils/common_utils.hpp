// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>
#include <vector>

namespace CommonTestUtils {

template<typename vecElementType>
inline std::string vec2str(const std::vector<vecElementType> &vec) {
    if (vec.empty())
        return "()";

    std::ostringstream result;
    result << "(";
    std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<vecElementType>(result, "."));
    result << vec.back() << ")";
    return result.str();
}

template<typename vecElementType>
inline std::string vec2str(const std::vector<std::vector<vecElementType>> &vec) {
    std::ostringstream result;
    for (const auto &v : vec) {
        result << vec2str<vecElementType>(v);
    }
    return result.str();
}

}  // namespace CommonTestUtils
