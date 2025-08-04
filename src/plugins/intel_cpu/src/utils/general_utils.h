// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_shape.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

#if defined(__clang__) || defined(__GNUC__)
#    define OV_CPU_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#    define OV_CPU_FUNCTION_NAME __FUNCSIG__
#else
// Fallback
#    define OV_CPU_FUNCTION_NAME __func__
#endif

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline T rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename... Args>
constexpr bool any_of(T val, Args... items) {
    static_assert(sizeof...(Args) > 0, "'any_of' requires at least one item to compare against.");
    return ((val == items) || ...);
}

template <typename T, typename... Args>
constexpr bool none_of(T val, Args... items) {
    static_assert(sizeof...(Args) > 0, "'none_of' requires at least one item to compare against.");
    return !any_of(val, items...);
}

template <typename T, typename... Args>
constexpr bool all_of(T val, Args... items) {
    static_assert(sizeof...(Args) > 0, "'all_of' requires at least one item to compare against.");
    return ((val == items) && ...);
}

constexpr bool implication(bool cause, bool cond) {
    return !cause || !!cond;
}

#ifdef __cpp_lib_make_unique
using std::make_unique;
#else
template <class T, class... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

template <typename T>
std::string vec2str(const std::vector<T>& vec) {
    if (!vec.empty()) {
        std::ostringstream result;
        result << "(";
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(result, "."));
        result << vec.back() << ")";
        return result.str();
    }
    return "()";
}

/**
 * @brief Compares that two dims are equal and defined
 * @param lhs
 * first dim
 * @param rhs
 * second dim
 * @return result of comparison
 */
inline bool dimsEqualStrong(size_t lhs, size_t rhs) {
    return (lhs == rhs && lhs != Shape::UNDEFINED_DIM && rhs != Shape::UNDEFINED_DIM);
}

/**
 * @brief Compares that two shapes are equal
 * @param lhs
 * first shape
 * @param rhs
 * second shape
 * @return result of comparison
 */
inline bool dimsEqualStrong(const std::vector<size_t>& lhs,
                            const std::vector<size_t>& rhs,
                            size_t skipAxis = Shape::UNDEFINED_DIM) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); i++) {
        if (i != skipAxis && !dimsEqualStrong(lhs[i], rhs[i])) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Compares that two dims are equal or undefined
 * @param lhs
 * first dim
 * @param rhs
 * second dim
 * @return result of comparison
 */
inline bool dimsEqualWeak(size_t lhs, size_t rhs) {
    return (lhs == Shape::UNDEFINED_DIM || rhs == Shape::UNDEFINED_DIM || lhs == rhs);
}

/**
 * @brief Compares that two shapes are equal or undefined
 * @param lhs
 * first shape
 * @param rhs
 * second shape
 * @param skipAxis
 * marks shape axis which shouldn't be validated
 * @return result of comparison
 */
inline bool dimsEqualWeak(const std::vector<size_t>& lhs,
                          const std::vector<size_t>& rhs,
                          size_t skipAxis = Shape::UNDEFINED_DIM) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); i++) {
        if (i != skipAxis && !dimsEqualWeak(lhs[i], rhs[i])) {
            return false;
        }
    }

    return true;
}

inline ov::element::Type getMaxPrecision(std::vector<ov::element::Type> precisions) {
    if (!precisions.empty()) {
        return *std::max_element(precisions.begin(),
                                 precisions.end(),
                                 [](const ov::element::Type& lhs, const ov::element::Type& rhs) {
                                     return lhs.size() > rhs.size();
                                 });
    }

    return ov::element::dynamic;
}

inline std::vector<std::string> split(const std::string& str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elements;
    while (std::getline(ss, item, delim)) {
        elements.emplace_back(item);
    }
    return elements;
}

template <class Container>
inline std::string join(const Container& strs, char delim) {
    if (strs.empty()) {
        return {};
    }

    std::stringstream result;
    auto it = strs.begin();
    result << *it++;
    for (; it != strs.end(); it++) {
        result << delim << *it;
    }
    return result.str();
}

template <typename Container, typename T>
inline bool any_of_values(const Container& container, const T& value) {
    return std::find(container.begin(), container.end(), value) != container.end();
}

template <typename Container, typename T>
inline bool all_of_values(const Container& container, const T& value) {
    return std::all_of(container.begin(), container.end(), [&](const auto& elem) {
        return elem == value;
    });
}

template <typename T>
inline bool contains(const std::vector<T>& v, const T& value) {
    return std::any_of(v.begin(), v.end(), [&](const auto& elem) {
        return elem == value;
    });
}

template <class Map>
bool contains_key_value(const Map& m, const typename Map::value_type& kv) {
    const auto& [k, v] = kv;
    if (auto it = m.find(k); it != m.end()) {
        return it->second == v;
    }

    return false;
}

}  // namespace ov::intel_cpu
