// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace ov {
namespace util {

template <typename T>
std::string join(const T& v, const std::string& sep = ", ") {
    std::ostringstream ss;
    size_t count = 0;
    for (const auto& x : v) {
        if (count++ > 0) {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}

template <typename T>
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << ov::util::join(v) << " ]";
    return os.str();
}

std::string to_lower(const std::string& s);

std::string to_upper(const std::string& s);

std::string trim(const std::string& s);

std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

template <typename T>
T ceil_div(const T& x, const T& y) {
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}
}  // namespace util
}  // namespace ov
