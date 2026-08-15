// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local exception layer. The standalone Vulkan core has no dependency on
// openvino/core; this header provides the ov::Exception / OPENVINO_ASSERT /
// OPENVINO_THROW surface that the runtime interfaces were written against.

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

namespace ov {

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

inline std::string build_message() {
    return {};
}

template <typename T, typename... Args>
std::string build_message(const T& first, const Args&... rest) {
    std::ostringstream ss;
    ss << first;
    return ss.str() + build_message(rest...);
}

}  // namespace detail
}  // namespace ov

#define OPENVINO_THROW(...) throw ::ov::Exception(::ov::detail::build_message(__VA_ARGS__))

#define OPENVINO_ASSERT(condition, ...)                                  \
    do {                                                                 \
        if (!(condition)) {                                              \
            OPENVINO_THROW(__VA_ARGS__);                                 \
        }                                                                \
    } while (0)
