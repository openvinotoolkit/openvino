// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <inference_engine.hpp>

namespace MKLDNNPlugin {

template<typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template<typename T, typename U>
inline T rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename T, typename P>
constexpr bool everyone_is(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

constexpr inline bool implication(bool cause, bool cond) {
    return !cause || !!cond;
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

inline std::string getExceptionDescWithoutStatus(const InferenceEngine::Exception& ex) {
    std::string desc = ex.what();
    IE_SUPPRESS_DEPRECATED_START
    if (ex.getStatus() != 0) {
        size_t pos = desc.find("]");
        if (pos != std::string::npos) {
            if (desc.size() == pos + 1) {
                desc.erase(0, pos + 1);
            } else {
                desc.erase(0, pos + 2);
            }
        }
    }
    IE_SUPPRESS_DEPRECATED_END

    return desc;
}

template<typename T>
std::string vec2str(const std::vector<T> &vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return std::string("()");
}

}  // namespace MKLDNNPlugin